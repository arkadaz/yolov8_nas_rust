use ndarray::prelude::*;
use opencv::{core, highgui, prelude::*, videoio, Result};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, SessionBuilder, Value};
use std::{sync::Arc, vec};

const YOLO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
fn main() -> Result<(), Box<dyn std::error::Error>> {
    //init ONNX
    tracing_subscriber::fmt::init();
    let status = ExecutionProvider::CUDA(Default::default()).is_available();
    println!("CUDA STATUS {:?}", status);
    let environment = Arc::new(
        Environment::builder()
            .with_name("YOLOV8")
            .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
            .build()?
            .into_arc(),
    );
    // let model_name = "yolov8n.onnx";
    let model_name = "yolo_nas_s.onnx";
    let model = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(12)?
        .with_model_from_file(model_name)
        .unwrap();
    //opencv
    let window = "video capture";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?; // 0 is the default camera
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    loop {
        let mut image = Mat::default();
        cam.read(&mut image)?;
        let mut image_bgr640 = Mat::default();
        let resize_to_640 = opencv::core::Size::new(640, 640);
        opencv::imgproc::resize(
            &image,
            &mut image_bgr640,
            resize_to_640,
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;
        let mut image_rgb640 = Mat::default();
        opencv::imgproc::cvt_color(
            &image_bgr640,
            &mut image_rgb640,
            opencv::imgproc::COLOR_BGR2RGB,
            0,
        )?;
        let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();
        for i in 0..image_rgb640.size()?.width as usize {
            for j in 0..image_rgb640.size()?.height as usize {
                let pixel: &core::VecN<u8, 3> = image_rgb640
                    .at_2d::<core::Vec3b>(i as i32, j as i32)
                    .unwrap();
                input[[0, 0, i, j]] = (pixel[0] as f32) / 255.0;
                input[[0, 1, i, j]] = (pixel[1] as f32) / 255.0;
                input[[0, 2, i, j]] = (pixel[2] as f32) / 255.0;
            }
        }
        let input_as_values = &input.as_standard_layout();
        let model_inputs = vec![Value::from_array(model.allocator(), input_as_values).unwrap()];
        let outputs = model.run(model_inputs)?;

        let results = if model_name == "yolov8.onnx" {
            let output = outputs
                .get(0)
                .unwrap()
                .try_extract::<f32>()
                .unwrap()
                .view()
                .t()
                .into_owned();
            let results = process_output_yolov8(output, 640, 640);
            results
        } else {
            let output_bbox = outputs
                .get(0)
                .unwrap()
                .try_extract::<f32>()
                .unwrap()
                .view()
                .t()
                .into_owned();
            let output_class = outputs
                .get(1)
                .unwrap()
                .try_extract::<f32>()
                .unwrap()
                .view()
                .t()
                .into_owned();

            let mut output =
                ndarray::concatenate(Axis(0), &[output_bbox.view(), output_class.view()]).unwrap();
            output.swap_axes(0, 1);
            let results = process_output_yolonas(output, 640, 640);
            results
        };

        for result in results {
            let rec = opencv::core::Rect::new(
                result.0 as i32,
                result.1 as i32,
                (result.2 - result.0) as i32,
                (result.3 - result.1) as i32,
            );
            let color = opencv::core::Scalar::new(1.0, 2.0, 3.0, 4.0);
            opencv::imgproc::rectangle(&mut image_bgr640, rec, color, 5, 0, 0)?;
            let text_point =
                opencv::core::Point::new((result.0 - 5.0) as i32, (result.1 - 5.0) as i32);
            opencv::imgproc::put_text(
                &mut image_bgr640,
                result.4,
                text_point,
                1,
                1.5,
                color,
                2,
                0,
                false,
            )?;
            println!("{:?}", result);
        }
        if image.size()?.width > 0 {
            highgui::imshow(window, &image_bgr640)?;
        }
        let key = highgui::wait_key(1)?;
        if key > 0 && key != 255 {
            break;
        }
    }
    Ok(())
}

fn iou(
    box1: &(f32, f32, f32, f32, &'static str, f32),
    box2: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
fn union(
    box1: &(f32, f32, f32, f32, &'static str, f32),
    box2: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _) = *box2;
    let box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    let box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
fn intersection(
    box1: &(f32, f32, f32, f32, &'static str, f32),
    box2: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2 - x1) * (y2 - y1);
}

fn process_output_yolov8(
    output: Array<f32, IxDyn>,
    img_width: u32,
    img_height: u32,
) -> Vec<(f32, f32, f32, f32, &'static str, f32)> {
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.5 {
            continue;
        }
        let label = YOLO_CLASSES[class_id];
        let xc = row[0] / 640.0 * (img_width as f32);
        let yc = row[1] / 640.0 * (img_height as f32);
        let w = row[2] / 640.0 * (img_width as f32);
        let h = row[3] / 640.0 * (img_height as f32);
        let x1 = xc - w / 2.0;
        let x2 = xc + w / 2.0;
        let y1 = yc - h / 2.0;
        let y2 = yc + h / 2.0;
        boxes.push((x1, y1, x2, y2, label, prob));
    }

    boxes.sort_by(|box1, box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len() > 0 {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| iou(&boxes[0], box1) < 0.7)
            .map(|x| *x)
            .collect()
    }
    return result;
}

fn process_output_yolonas(
    output: Array<f32, IxDyn>,
    img_width: u32,
    img_height: u32,
) -> Vec<(f32, f32, f32, f32, &'static str, f32)> {
    let mut boxes = Vec::new();
    let output = output.slice(s![.., .., 0]);
    for row in output.axis_iter(Axis(0)) {
        let row: Vec<_> = row.iter().map(|x| *x).collect();
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();
        if prob < 0.5 {
            continue;
        }
        let label = YOLO_CLASSES[class_id];
        let xc = row[0] / 640.0 * (img_width as f32);
        let yc = row[1] / 640.0 * (img_height as f32);
        let w = row[2] / 640.0 * (img_width as f32);
        let h = row[3] / 640.0 * (img_height as f32);
        let x1 = xc;
        let x2 = w;
        let y1 = yc;
        let y2 = h;
        boxes.push((x1, y1, x2, y2, label, prob));
    }

    boxes.sort_by(|box1, box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len() > 0 {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| iou(&boxes[0], box1) < 0.7)
            .map(|x| *x)
            .collect()
    }
    return result;
}
