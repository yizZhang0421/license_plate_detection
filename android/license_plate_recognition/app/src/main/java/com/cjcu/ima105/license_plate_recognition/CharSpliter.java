package com.cjcu.ima105.license_plate_recognition;

import android.service.autofill.SaveCallback;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CharSpliter {
    private static Mat increase_brightness(Mat img,int value){ //30
        Mat hsv = new Mat();
        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_BGR2HSV);
        List<Mat> h_s_v = new ArrayList<>();
        Core.split(hsv, h_s_v);
        Mat v = h_s_v.get(2);
        Core.add(v, new Mat(v.height(), v.width(), CvType.CV_8U, Scalar.all(value)), v);
        h_s_v.set(2, v);
        Mat final_hsv = new Mat();
        Core.merge(h_s_v, final_hsv);
        Imgproc.cvtColor(final_hsv, img, Imgproc.COLOR_HSV2BGR);
        return img;
    }
    private static Mat deskew(Mat binary_im, int max_skew){ //10
        int height = binary_im.height();
        int width= binary_im.width();
        Mat lines = new Mat();
        Imgproc.HoughLinesP(binary_im, lines, 1, Math.PI/180, 200, width/12, width/150);
        if(lines.empty()){
            return binary_im;
        }
        ArrayList<Double> angles = new ArrayList<>();
        for(int r=0;r<lines.rows();r++){
            double[] x1_y1_x2_y2 = lines.get(r, 0);
            double x1=x1_y1_x2_y2[0];
            double y1=x1_y1_x2_y2[1];
            double x2=x1_y1_x2_y2[2];
            double y2=x1_y1_x2_y2[3];
            angles.add(Math.atan2(y2 - y1, x2 - x1));
        }
        int count=0;
        for(double angle : angles){
            if(Math.abs(angle)>Math.PI/4){
                count++;
            }
        }
        boolean landscape=count>angles.size()/2;
        ArrayList<Double> new_angles = new ArrayList<>();
        if(landscape){
            for(int i=0;i<angles.size();i++){
                if(Math.toRadians(90-max_skew)<Math.abs(angles.get(i)) && Math.abs(angles.get(i))<Math.toRadians(90+max_skew)){
                    new_angles.add(angles.get(i));
                }
            }
        }
        else{
            for(int i=0;i<angles.size();i++){
                if(Math.abs(angles.get(i))<Math.toRadians(max_skew)){
                    new_angles.add(angles.get(i));
                }
            }
        }
        if(new_angles.size()<5){
            return binary_im;
        }
        Collections.sort(new_angles);
        double angle_deg = Math.toDegrees(new_angles.get(0)+((new_angles.get(new_angles.size()-1)-new_angles.get(0))/(double)2));
        if (landscape){
            if(angle_deg<0){
                Core.rotate(binary_im, binary_im, Core.ROTATE_90_CLOCKWISE);
                angle_deg += 90;
            }
            else if(angle_deg>0){
                Core.rotate(binary_im, binary_im, Core.ROTATE_90_COUNTERCLOCKWISE);
                angle_deg -= 90;
            }
        }
        Mat M = Imgproc.getRotationMatrix2D(new Point(width/2, height/2), angle_deg, 1);
        Imgproc.warpAffine(binary_im, binary_im, M, new Size(width, height), Imgproc.WARP_FILL_OUTLIERS);
        return binary_im;
    }
    private static float IOU(Rect origin, Rect of_origin) {
        if (origin.x > of_origin.x+of_origin.width) return 0.0f;
        if (origin.y > of_origin.y+of_origin.height) return 0.0f;
        if (origin.x+origin.width < of_origin.x) return 0.0f;
        if (origin.y+origin.height < of_origin.y) return 0.0f;
        int colInt =  Math.min(origin.x+origin.width,of_origin.x+of_origin.width) - Math.max(origin.x, of_origin.x);
        int rowInt =  Math.min(origin.y+origin.height,of_origin.y+of_origin.height) - Math.max(origin.y,of_origin.y);
        int intersection = colInt * rowInt;
        int area1 = origin.width*origin.height;
        int area2 = of_origin.width*of_origin.height;
        return (float)intersection / (float)(area1 + area2 - intersection);
    }
    private static float max_IOU(Row row, Rect new_box) {
        float max_iou=-9999f;
        for(Rect m : row.member) {
            float iou=IOU(m, new_box);
            if(iou>max_iou) {
                max_iou=iou;
            }
        }
        return max_iou;
    }
    public static ArrayList<Mat> get_char_binary_mat(Mat origin_mat){
        origin_mat=increase_brightness(origin_mat, 30);
        Imgproc.resize(origin_mat, origin_mat, new Size(366, (int)(Math.round(origin_mat.height()*366/origin_mat.width()))));
        Photo.fastNlMeansDenoising(origin_mat, origin_mat, 35, 7, 21);
//        Imgproc.erode(origin_mat, origin_mat, Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(1,5)));
        Imgproc.cvtColor(origin_mat, origin_mat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(origin_mat, origin_mat, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        origin_mat=deskew(origin_mat, 10);

        ArrayList<Row> same_row = new ArrayList<>();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(origin_mat, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);


        MainActivity.test=origin_mat.clone();
        for(int i=0;i<contours.size();i++) {
            MatOfPoint contour = contours.get(i);
            Rect x_y_w_h = Imgproc.boundingRect(contour);
            Imgproc.rectangle(MainActivity.test, new Point(x_y_w_h.x, x_y_w_h.y), new Point(x_y_w_h.x + x_y_w_h.width - 1, x_y_w_h.y + x_y_w_h.height - 1), new Scalar(0, 255, 0));
        }


        int threshold=10;
        for(int i=0;i<contours.size();i++){
            MatOfPoint contour=contours.get(i);
            Rect x_y_w_h = Imgproc.boundingRect(contour);
            int x=x_y_w_h.x;
            int y=x_y_w_h.y;
            int w=x_y_w_h.width;
            int h=x_y_w_h.height;
            boolean finded=false;
            if(x==0 || x+w-1==origin_mat.width()-1 || y==0 || y+h-1==origin_mat.height()-1){
                continue;
            }
            for(Row row : same_row){
                if(y>row.min_y_top && y<=row.min_y_bottom && y+h-1>=row.max_y_top && y+h-1<=row.max_y_bottom && max_IOU(row, new Rect(x,y,w,h))<0.3){
                    Mat mask = new Mat(origin_mat.height(), origin_mat.width(), CvType.CV_8U, Scalar.all(255));
                    Imgproc.drawContours(mask, contours, i, new Scalar(0), -1);
                    row.member.add(x_y_w_h);
                    row.mask.add(mask);
                    row.min_y_top=Math.min(row.min_y_top, y-threshold);
                    row.min_y_bottom=Math.max(row.min_y_bottom, y+threshold);
                    row.max_y_top=Math.min(row.max_y_top, (y+h-1)-threshold);
                    row.max_y_bottom=Math.max(row.max_y_bottom, (y+h-1)+threshold);
                    finded=true;
                    break;
                }
            }
            if(finded==false){
                Row row = new Row();
                row.min_y_top=y-threshold;
                row.min_y_bottom=y+threshold;
                row.max_y_top=y+h-1-threshold;
                row.max_y_bottom=y+h-1+threshold;
                Mat mask = new Mat(origin_mat.height(), origin_mat.width(), CvType.CV_8U, Scalar.all(255));
                Imgproc.drawContours(mask, contours, i, new Scalar(0), -1);
                row.member.add(x_y_w_h);
                row.mask.add(mask);
                same_row.add(row);
            }
        }
        Row target_row=same_row.get(0);
        for(Row row : same_row){
            if (row.member.size()>target_row.member.size() && origin_mat.height()/2>=row.min_y_top && origin_mat.height()/2<=row.max_y_bottom){
                target_row=row;
            }
        }
        for(int i=0;i<target_row.member.size();i++) {
            for(int j=i+1;j<target_row.member.size();j++) {
                if(target_row.member.get(j).x<target_row.member.get(i).x){
                    Rect tmp = target_row.member.get(j);
                    target_row.member.set(j, target_row.member.get(i));
                    target_row.member.set(i, tmp);

                    Mat tmp_mat = target_row.mask.get(j);
                    target_row.mask.set(j, target_row.mask.get(i));
                    target_row.mask.set(i, tmp_mat);
                }
            }
        }
        ArrayList<Mat> result = new ArrayList<>();
        for(int i=0;i<target_row.member.size();i++){
            Rect rect = target_row.member.get(i);
            Mat mask = target_row.mask.get(i);
            Mat char_in_binary=origin_mat.clone();
            Imgproc.threshold(char_in_binary, char_in_binary, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
            Core.add(char_in_binary, mask, char_in_binary);
            Imgproc.threshold(char_in_binary, char_in_binary, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
            Mat char_mat = new Mat(char_in_binary, rect);
            Imgproc.erode(char_mat, char_mat, Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(3,5)));
            Imgproc.resize(char_mat, char_mat, new Size(64, 64));
            char_mat.convertTo(char_mat, CvType.CV_64F, 1.0D/255.0D);
            result.add(char_mat);
        }
        return result;
    }

    static class Row{
        int min_y_top;
        int min_y_bottom;
        int max_y_top;
        int max_y_bottom;
        ArrayList<Rect> member;
        ArrayList<Mat> mask;
        public Row(){
            this.member=new ArrayList<>();
            this.mask=new ArrayList<>();
        }
    }


}
