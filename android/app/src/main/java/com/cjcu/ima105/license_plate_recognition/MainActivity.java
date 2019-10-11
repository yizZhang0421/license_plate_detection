package com.cjcu.ima105.license_plate_recognition;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import com.camerakit.CameraKitView;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.appcompat.widget.LinearLayoutCompat;


public class MainActivity extends AppCompatActivity {
    static {OpenCVLoader.initDebug();}

    private Net net;
    private String[] classNames;
    private CameraKitView cameraKitView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // download darknet file
        boolean[] darknet_require = new boolean[]{false, false};
        for(String file : this.getFilesDir().list()){
            if(file.equals("lp.cfg")){
                darknet_require[0]=true;
            }
            if(file.equals("lp.weights")){
                darknet_require[1]=true;
            }
        }
        if(darknet_require[0]==false){
            assetToInternal("lp.cfg");
        }
        if(darknet_require[1]==false){
            assetToInternal("lp.weights");
        }
        net = Dnn.readNetFromDarknet(this.getFilesDir().getPath()+"/lp.cfg", this.getFilesDir().getPath()+"/lp.weights");

        // setup camera
        cameraKitView = findViewById(R.id.camera);
        RelativeLayout button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraKitView.captureImage(new CameraKitView.ImageCallback() {
                    @Override
                    public void onImage(CameraKitView cameraKitView, byte[] bytes) {
                        Log.e("asd", bytes.length+"");
                        Mat frame = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.CV_LOAD_IMAGE_COLOR);
                        Mat blob = Dnn.blobFromImage(frame, 0.00392156862745098f, new Size(224, 224), new Scalar(0, 0, 0), false, false);
                        net.setInput(blob);
                        Mat detections = net.forward();

                        int width = frame.width();
                        int height = frame.height();
                        ArrayList<Integer> class_ids = new ArrayList<>();
                        ArrayList<Double> confidences = new ArrayList<>();
                        ArrayList<int[]> boxes = new ArrayList<>();
                        for (int r = 0; r < detections.rows(); r++) {
                            ArrayList<Double> scores = new ArrayList<>();
                            for (int i = 5; i < detections.cols(); i++) {
                                scores.add(detections.get(r, i)[0]);
                            }
                            int class_id = 0;
                            double confidence = scores.get(class_id);
                            if (confidence > 0.5) {
                                int center_x = (int) (detections.get(r, 0)[0] * width);
                                int center_y = (int) (detections.get(r, 1)[0] * height);
                                int w = (int) (detections.get(r, 2)[0] * width);
                                int h = (int) (detections.get(r, 3)[0] * height);
                                int x = (int) (center_x - w / 2);
                                int y = (int) (center_y - h / 2);
                                boxes.add(new int[]{x, y, w, h});
                                confidences.add(confidence);
                                class_ids.add(class_id);
                            }
                        }
                        LinearLayout info_area = findViewById(R.id.info_area);
                        info_area.removeAllViews();
                        for (int[] box : boxes) {
                            int minx=box[0];
                            int miny=box[1];
                            int maxx=box[0]+box[2]-1;
                            int maxy=box[1]+box[3]-1;
                            if(minx-10>0){
                                minx-=10;
                            }
                            else{
                                minx=0;
                            }
                            if(miny-10>0){
                                miny-=10;
                            }
                            else{
                                miny=0;
                            }
                            if(maxx+10<frame.width()){
                                maxx+=10;
                            }
                            else{
                                maxx=frame.width()-1;
                            }
                            if(maxy+10<frame.height()){
                                maxy+=10;
                            }
                            else{
                                maxy=frame.height()-1;
                            }

                            Log.e("asd", minx+" "+miny+" "+(maxx-minx+1)+" "+(maxy-miny+1));

//                            Imgproc.rectangle(frame, new Point(box[0], box[1]), new Point(box[0] + box[2] - 1, box[1] + box[3] - 1), new Scalar(0, 255, 0));
                            Mat crop = new Mat(frame, new Rect(minx, miny, (maxx-minx+1), (maxy-miny+1)));


                            ArrayList<Mat> chars = CharSpliter.get_char_binary_mat(crop.clone()); //testttttt



                            Size size= new Size((int)Math.round((float)(maxx-minx+1)*200.0/(float)(maxy-miny+1)),200);
                            Imgproc.resize(crop, crop, size);
                            Imgproc.cvtColor(crop, crop, Imgproc.COLOR_BGR2RGB);


                            crop = chars.get(0); //testttttt
                            Imgproc.resize(crop, crop, size); //testttttt
                            Imgproc.cvtColor(crop, crop, Imgproc.COLOR_GRAY2RGB); //testttttt


                            Bitmap bmp = null;
                            try {
                                bmp = Bitmap.createBitmap(crop.cols(), crop.rows(), Bitmap.Config.RGB_565);
                                Utils.matToBitmap(crop, bmp);
                            }catch (CvException e){Log.d("Exception",e.getMessage());}
                            AppCompatImageView imageView = new AppCompatImageView(MainActivity.this);
                            imageView.setImageBitmap(bmp);
                            info_area.addView(imageView);
                        }
                    }
                });
            }
        });
        classNames=new String[]{"License_Plate"};
    }
    @Override
    protected void onStart() {
        super.onStart();
        cameraKitView.onStart();
    }
    @Override
    protected void onResume() {
        super.onResume();
        cameraKitView.onResume();
    }
    @Override
    protected void onPause() {
        cameraKitView.onPause();
        super.onPause();
    }
    @Override
    protected void onStop() {
        cameraKitView.onStop();
        super.onStop();
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        cameraKitView.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private void assetToInternal(String file) {
        AssetManager assetManager = getAssets();
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(file);
            File outFile = new File(this.getFilesDir().getPath(), file);
            out = new FileOutputStream(outFile);
            byte[] buffer = new byte[1024];
            int read;
            while((read = in.read(buffer)) != -1){
                out.write(buffer, 0, read);
            }
            in.close();
            in = null;
            out.flush();
            out.close();
            out = null;
        } catch(IOException e) {
            Log.e("tag", "Failed to copy asset file: " + file, e);
        }
    }
}
