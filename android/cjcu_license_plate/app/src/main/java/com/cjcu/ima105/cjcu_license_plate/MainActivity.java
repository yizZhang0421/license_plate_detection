package com.cjcu.ima105.cjcu_license_plate;

import androidx.annotation.MainThread;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
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
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    static {OpenCVLoader.initDebug();}
    public static Mat test;

    private static Interpreter tflite;

    private CameraBridgeViewBase mOpenCvCameraView;
    private LinearLayout info_area;
    private RelativeLayout button;
    private static boolean click;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            tflite = new Interpreter(loadModelFile());
        }catch (Exception e){
            Log.e("asd", "load tflite fail");
        }

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        info_area=findViewById(R.id.info_area);
        button=findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(click==false) {
                    click = true;
                }
            }
        });
    }

    @Override
    public void onCameraViewStarted(int width, int height) {}

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        int CROP_WIDTH=300;
        int CROP_HEIGHT=144;
        final int minx=(frame.width()/2)-(CROP_WIDTH/2);
        final int miny=(frame.height()/2)-(CROP_HEIGHT/2);
        final int maxx=(frame.width()/2)+(CROP_WIDTH/2);
        final int maxy=(frame.height()/2)+(CROP_HEIGHT/2);
        final Mat origin = frame.clone();
        Imgproc.rectangle(frame,
                new Point(minx, miny),
                new Point(maxx, maxy),
                new Scalar(0, 255, 0), 5);
        if(MainActivity.this.click){
            MainActivity.this.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    info_area.removeAllViews();
                }
            });
            Mat crop = new Mat(origin, new Rect(minx, miny, maxx-minx+1, maxy-miny+1));
            final Bitmap bmp = Bitmap.createBitmap(crop.cols(), crop.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(crop, bmp);
            MainActivity.this.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    AppCompatImageView imageView = new AppCompatImageView(MainActivity.this);
                    imageView.setImageBitmap(bmp);
                    info_area.addView(imageView);
                }
            });
            ArrayList<Mat> chars = CharSpliter.get_char_binary_mat(crop);
            crop.release();
            Log.e("asd", chars.size()+"");
            String result="";
            for(Mat char_mat : chars){
                ByteBuffer imgData = ByteBuffer.allocateDirect(1* 64 * 64 * 1 * 4);
                imgData.order(ByteOrder.nativeOrder());
                float[][] ProbArray = new float[1][35];
                imgData.rewind();
                for (int r = 0; r < 64; r++) {
                    for (int c = 0; c < 64; c++) {
                        imgData.putFloat((float)char_mat.get(r,c)[0]);
                    }
                }
                tflite.run(imgData, ProbArray);
                result += maxProbIndex(ProbArray[0]);
                char_mat.release();
            }
            final String finalResult = result;
            MainActivity.this.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    AppCompatTextView textView = new AppCompatTextView(MainActivity.this);
                    textView.setText(finalResult);
                    textView.setTextSize(TypedValue.COMPLEX_UNIT_PX, 50);
                    info_area.addView(textView);
                }
            });
        }
        MainActivity.this.click=false;
        origin.release();
        return frame;
    }

    @Override
    public void onCameraViewStopped() {}



    @Override
    protected void onStart() {
        super.onStart();
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA},0);
        }
        else{
            mOpenCvCameraView.enableView();
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode==0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
            mOpenCvCameraView.enableView();
        }
        else{
            Toast.makeText(this, "cannot open camera.", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("lp_char.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private String maxProbIndex(float[] probs) {
        int maxIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        String result ="";
        switch (maxIndex){
            case 0:result="0";break;
            case 1:result="1";break;
            case 2:result="2";break;
            case 3:result="3";break;
            case 4:result="4";break;
            case 5:result="5";break;
            case 6:result="6";break;
            case 7:result="7";break;
            case 8:result="8";break;
            case 9:result="9";break;
            case 10:result="A";break;
            case 11:result="B";break;
            case 12:result="C";break;
            case 13:result="D";break;
            case 14:result="E";break;
            case 15:result="F";break;
            case 16:result="G";break;
            case 17:result="H";break;
            case 18:result="I";break;
            case 19:result="J";break;
            case 20:result="K";break;
            case 21:result="L";break;
            case 22:result="M";break;
            case 23:result="N";break;
            case 24:result="P";break;
            case 25:result="Q";break;
            case 26:result="R";break;
            case 27:result="S";break;
            case 28:result="T";break;
            case 29:result="U";break;
            case 30:result="V";break;
            case 31:result="W";break;
            case 32:result="X";break;
            case 33:result="Y";break;
            case 34:result="Z";break;
        }
        return result;
    }
}
