package com.example.resistorreader;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.content.Context;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

public class ResistorPredictor {
    private final Context context;
    private final OrtEnvironment env;
    private OrtSession session;

    public ResistorPredictor(Context context) {
        this.context = context;
        this.env = OrtEnvironment.getEnvironment();
        initializeOnnxSession();
    }

    // Method to initialize the ONNX model session
    private void initializeOnnxSession() {
        try {
            // เปิดไฟล์โมเดลจาก assets
            InputStream inputStream = context.getAssets().open("resistor_model.onnx");

            // อ่านไฟล์เป็น byte array
            byte[] modelBytes = new byte[inputStream.available()];
            inputStream.read(modelBytes);
            inputStream.close();

            // สร้างเซสชันจาก byte array
            session = env.createSession(modelBytes, new OrtSession.SessionOptions());

        } catch (IOException | OrtException e) {
            e.printStackTrace();
            session = null; // ตั้งค่า session ให้เป็น null หากมีข้อผิดพลาด
        }
    }

    // Method to predict the resistance based on input values
    public float predict(float hue, float saturation, float value) throws OrtException {
        if (session == null) {
            throw new IllegalStateException("ONNX model session is not initialized.");
        }

        float[] inputArray = new float[]{hue, saturation, value};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputArray), new long[]{1, 3});

        OrtSession.Result output = session.run(Collections.singletonMap("float_input", inputTensor));
        Object outputValue = output.get(0).getValue();

        float result;
        if (outputValue instanceof long[]) {
            long[] longArray = (long[]) outputValue;
            result = (float) longArray[0]; // แปลง long เป็น float
        } else if (outputValue instanceof float[][]) {
            result = ((float[][]) outputValue)[0][0];
        } else {
            throw new OrtException("Unexpected output type: " + outputValue.getClass());
        }

        return result;
    }

    // Method to check if session is initialized
    public boolean isInitialized() {
        return session != null;
    }

    // Method to close the session and environment
    public void close() {
        try {
            if (session != null) {
                session.close();
            }
            env.close();
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
}
