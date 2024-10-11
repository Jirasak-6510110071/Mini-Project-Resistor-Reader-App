package com.example.resistorreader;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import ai.onnxruntime.OrtException;
import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Paint;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {
    private ResistorPredictor resistorPredictor;
    CameraBridgeViewBase cameraBridgeViewBase;
    TextView hsvTextView, predictionText;
    Paint paint;
    int frameWidth = 80; // กว้างกรอบ
    int frameHeight = 40; // สูงกรอบ

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        hsvTextView = findViewById(R.id.hsvValueText);
        cameraBridgeViewBase = findViewById(R.id.cameraView);
        predictionText = findViewById(R.id.predictionText);


        // ตั้งค่า Paint สำหรับวาดกรอบ
        paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5);

        getPermission();
        try {
            resistorPredictor = new ResistorPredictor(this);
            if (!resistorPredictor.isInitialized()) {
                Log.e("MainActivity", "ResistorPredictor is not initialized after creation");
            }
        } catch (Exception e) {
            Log.e("MainActivity", "Failed to initialize ResistorPredictor", e);
        }

        cameraBridgeViewBase.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
            @Override
            public void onCameraViewStarted(int width, int height) {}

            @Override
            public void onCameraViewStopped() {}

            @SuppressLint("DefaultLocale")
            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                Mat frame = inputFrame.rgba();
                int x = (frame.cols() - frameWidth) / 2; // ตำแหน่ง X ของกรอบ
                int y = (frame.rows() - frameHeight) / 2; // ตำแหน่ง Y ของกรอบ

                // วาดกรอบ
                drawRectangle(frame, x, y, frameWidth, frameHeight);

                // สร้าง croppedFrame จากกรอบ
                Mat croppedFrame = frame.submat(y, y + frameHeight, x, x + frameWidth);

                // คำนวณค่าเฉลี่ยของ HSV จาก croppedFrame
                double[] avgHSV = calculateAverageHSV(croppedFrame);

                // อัพเดต TextView ด้วยค่า HSV
                runOnUiThread(() -> {
                    if (avgHSV != null) {
                        hsvTextView.setText(String.format("Average HSV: H=%.2f, S=%.2f, V=%.2f",
                                avgHSV[0], avgHSV[1], avgHSV[2]));

                        // ส่งค่าเฉลี่ย HSV ไปยังโมเดลเพื่อทำนาย
                        try {
                            if (resistorPredictor != null && resistorPredictor.isInitialized()) {
                                // แปลง double เป็น float ก่อนที่จะส่ง
                                float h = (float) avgHSV[0];
                                float s = (float) avgHSV[1];
                                float v = (float) avgHSV[2];

                                float predictedResistance = resistorPredictor.predict(h, s, v);
                                predictionText.setText("Prediction: " + Float.toString(predictedResistance));
                            } else {
                                Log.e("MainActivity", "ResistorPredictor is not initialized");
                            }
                        } catch (OrtException e) {
                            Log.e("Prediction", "Prediction error: " + e.getMessage());
                        }
                    }
                });

                return frame;
            }
        });

        if (OpenCVLoader.initDebug()) {
            cameraBridgeViewBase.enableView();
        }
    }
    private void drawRectangle(Mat frame, int x, int y, int width, int height) {
        // วาดกรอบสี่เหลี่ยม
        Imgproc.rectangle(frame, new org.opencv.core.Point(x, y),
                new org.opencv.core.Point(x + width, y + height),
                new org.opencv.core.Scalar(255, 0, 0, 255), 2);
    }

    public double[] calculateAverageHSV(Mat croppedFrame) {
        List<double[]> hsvValues = new ArrayList<>();
        Mat hsvFrame = new Mat();
        Imgproc.cvtColor(croppedFrame, hsvFrame, Imgproc.COLOR_RGBA2BGR);
        Imgproc.cvtColor(hsvFrame, hsvFrame, Imgproc.COLOR_BGR2HSV);

        for (int row = 0; row < hsvFrame.rows(); row++) {
            for (int col = 0; col < hsvFrame.cols(); col++) {
                double[] hsvPixel = hsvFrame.get(row, col);
                hsvValues.add(hsvPixel);
            }
        }

        if (!hsvValues.isEmpty()) {
            double[] avgHSV = {0, 0, 0};
            for (double[] hsv : hsvValues) {
                avgHSV[0] += hsv[0];
                avgHSV[1] += hsv[1];
                avgHSV[2] += hsv[2];
            }
            avgHSV[0] /= hsvValues.size();
            avgHSV[1] /= hsvValues.size();
            avgHSV[2] /= hsvValues.size();
            return avgHSV;
        }
        return null;
    }

    void getPermission() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 101);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission();
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cameraBridgeViewBase);
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (resistorPredictor != null) {
            resistorPredictor.close();
        }
    }
}
