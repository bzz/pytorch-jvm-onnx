package org.jetbrains.ai.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Arrays;
import java.util.Map;

public class MainCLI {

    public static void main(String[] args) {
        var modelFile = "src/main/resources/model.onnx";
        System.out.println("Loading model from " + modelFile);

        var env = OrtEnvironment.getEnvironment();
        try {
            var session = env.createSession(modelFile, new OrtSession.SessionOptions());

            var inputArr = new float[1][103];
            // TODO(bzz): read from STDIN, convert libsvm -> float[]][
            OnnxTensor t1 = OnnxTensor.createTensor(env, inputArr);

            var inputs = Map.of("input.1", t1);
            var results = session.run(inputs);

            System.out.println("output (" + results.size() + "): " + results.get(0).getInfo());

            float[][] labels = (float[][]) results.get(0).getValue();
            System.out.println("output value: " + Arrays.toString(labels[0]));
        } catch (OrtException e) {
            e.printStackTrace();
            try {
                env.close();
            } catch (OrtException oe) {
                oe.printStackTrace();
            }
        }

    }
}
