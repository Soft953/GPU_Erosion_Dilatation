# pragma once

# include <opencv2/opencv.hpp>
# include <iostream>
# include <string>


class ImageHandler 
{
    public : 
        ImageHandler(char* path);

        void display_matrix(int** mat, int h, int w);

        void display_image();

        int* mat2Vec(cv::Mat mat);

        int** vec2Mat(int* vector, int h, int w);

        cv::Mat image_get();

        cv::Mat greyscale_get();

        cv::Mat binary_get();


        

    private : 

        char* path_;
        cv::Mat image_;
        cv::Mat greyscale_mat_;
        cv::Mat binary_mat_;
};