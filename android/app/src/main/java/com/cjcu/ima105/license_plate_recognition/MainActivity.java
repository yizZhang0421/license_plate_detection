package com.cjcu.ima105.license_plate_recognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import com.camerakit.CameraKitView;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.appcompat.widget.AppCompatTextView;


public class MainActivity extends AppCompatActivity {
    static {OpenCVLoader.initDebug();}

    private Net net;
    private Interpreter tflite;
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
        try {
            tflite = new Interpreter(loadModelFile());
        }catch (Exception e){
            Log.e("asd", "load tflite fail");
        }

        // setup camera
        cameraKitView = findViewById(R.id.camera);
        RelativeLayout button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraKitView.captureImage(new CameraKitView.ImageCallback() {
                    @Override
                    public void onImage(CameraKitView cameraKitView, byte[] bytes) {
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

                            String result="";
                            ArrayList<Mat> chars = CharSpliter.get_char_binary_mat(crop.clone());
                            Log.e("asd", chars.size()+"");
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
                            }

                            Size size= new Size((int)Math.round((float)(maxx-minx+1)*200.0/(float)(maxy-miny+1)),200);
                            Imgproc.resize(crop, crop, size);
                            Imgproc.cvtColor(crop, crop, Imgproc.COLOR_BGR2RGB);

                            Bitmap bmp = null;
                            try {
                                bmp = Bitmap.createBitmap(crop.cols(), crop.rows(), Bitmap.Config.RGB_565);
                                Utils.matToBitmap(crop, bmp);
                            }catch (CvException e){Log.d("Exception",e.getMessage());}
                            AppCompatImageView imageView = new AppCompatImageView(MainActivity.this);
                            imageView.setImageBitmap(bmp);
                            info_area.addView(imageView);
                            AppCompatTextView textView = new AppCompatTextView(MainActivity.this);
                            textView.setText(result);
                            textView.setTextSize(TypedValue.COMPLEX_UNIT_PX, 100);
                            info_area.addView(textView);
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
