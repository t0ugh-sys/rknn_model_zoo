#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "rknn_api.h"
#include "yolov8.h"
#include "postprocess.h"
#include "image_utils.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        printf("Usage: %s <rknn_model> <video_source>\n", argv[0]);
        printf("  video_source: camera id (e.g. 0) or video file path\n");
        printf("  Output: saved to output.mp4 + print per-frame detections\n");
        return -1;
    }

    const char* model_path   = argv[1];
    const char* video_source = argv[2];

    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    int ret = init_yolov8_model(model_path, &app_ctx);
    if (ret != 0) {
        printf("init_yolov8_model failed! ret=%d\n", ret);
        deinit_post_process();
        return -1;
    }

    cv::VideoCapture cap;
    if (strlen(video_source) == 1 && isdigit(video_source[0])) {
        cap.open(atoi(video_source), cv::CAP_V4L2);
    } else {
        cap.open(video_source);
    }
    if (!cap.isOpened()) {
        printf("Failed to open video source: %s\n", video_source);
        release_yolov8_model(&app_ctx);
        deinit_post_process();
        return -1;
    }

    int frame_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps_input = cap.get(cv::CAP_PROP_FPS);
    if (fps_input <= 0) fps_input = 30.0;

    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('H','2','6','4'),
                           fps_input, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        printf("Warning: VideoWriter failed to open, will not save video.\n");
    } else {
        printf("Saving inference result to output.mp4 (FPS: %.1f, Size: %dx%d)\n",
               fps_input, frame_width, frame_height);
    }

    cv::Mat frame;
    object_detect_result_list od_results;
    image_buffer_t src_image;

    long long frame_count = 0;

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        cap >> frame;
        if (frame.empty()) {
            printf("End of video. Total processed frames: %lld\n", frame_count);
            break;
        }

        frame_count++;

        cv::Mat display_img = frame.clone();

        // resize 到模型输入大小
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(app_ctx.model_width, app_ctx.model_height));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // 构造 image_buffer_t
        memset(&src_image, 0, sizeof(image_buffer_t));
        src_image.width     = resized.cols;
        src_image.height    = resized.rows;
        src_image.size      = resized.cols * resized.rows * 3;
        src_image.virt_addr = resized.data;
        src_image.format    = IMAGE_FORMAT_RGB888;

        // 推理
        ret = inference_yolov8_model(&app_ctx, &src_image, &od_results);
        if (ret != 0) {
            printf("inference failed on frame %lld! ret=%d\n", frame_count, ret);
            break;
        }

        // ------------------- 打印每帧检测结果（与官方 demo 一致） -------------------
        printf("Frame %lld detections (%d objects):\n", frame_count, od_results.count);
        for (int i = 0; i < od_results.count; ++i) {
            object_detect_result* det = &od_results.results[i];
            printf("  %s @ (%d %d %d %d) %.3f\n",
                   coco_cls_to_name(det->cls_id),
                   det->box.left, det->box.top,
                   det->box.right, det->box.bottom,
                   det->prop);
        }
        if (od_results.count == 0) {
            printf("  no objects detected\n");
        }
        // -------------------------------------------------------------------------

        // 将坐标缩放回原图并绘制
        float scale_x = (float)frame.cols / app_ctx.model_width;
        float scale_y = (float)frame.rows / app_ctx.model_height;

        char text[64];
        for (int i = 0; i < od_results.count; ++i) {
            object_detect_result* det = &od_results.results[i];

            int x1 = (int)(det->box.left   * scale_x);
            int y1 = (int)(det->box.top    * scale_y);
            int x2 = (int)(det->box.right  * scale_x);
            int y2 = (int)(det->box.bottom * scale_y);

            x1 = std::max(0, std::min(x1, frame.cols - 1));
            y1 = std::max(0, std::min(y1, frame.rows - 1));
            x2 = std::max(0, std::min(x2, frame.cols - 1));
            y2 = std::max(0, std::min(y2, frame.rows - 1));

            cv::rectangle(display_img, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(255, 0, 0), 3);

            sprintf(text, "%s %.1f%%", coco_cls_to_name(det->cls_id), det->prop * 100);
            cv::putText(display_img, text, cv::Point(x1, y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        // FPS 显示在视频中
        auto end = std::chrono::high_resolution_clock::now();
        double fps = 1000.0 / std::chrono::duration<double, std::milli>(end - start).count();
        sprintf(text, "FPS: %.1f  Frame: %lld", fps, frame_count);
        cv::putText(display_img, text, cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);

        // 保存帧
        if (writer.isOpened()) {
            writer << display_img;
        }
    }

    if (writer.isOpened()) {
        writer.release();
        printf("Video saved successfully: output.mp4\n");
    }

    release_yolov8_model(&app_ctx);
    deinit_post_process();
    cap.release();

    return 0;
}
