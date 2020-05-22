package org.pytorch.demo.nlp;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.Editable;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.util.Log;
import android.view.View;
import android.widget.EditText;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.appcompat.widget.Toolbar;

import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.InfoViewFactory;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.transformers.Feature;
import org.pytorch.demo.transformers.FeatureConverter;
import org.pytorch.demo.view.ResultRowView;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class NSMCTensorflowActivity extends BaseModuleActivity {
    private static final String TAG = "NSMCTensorflowDemo";
    private static final int NUM_LITE_THREADS = 4;
    private static final String MODEL_PATH = "nsmc_small.tflite";
    private static final String DIC_PATH = "nsmc_vocab.txt";

    private static final long EDIT_TEXT_STOP_DELAY = 600l;
    private static final String FORMAT_MS = "%dms";
    private static final String SCORES_FORMAT = "%.2f";

    private EditText mEditText;
    private View mResultContent;
    private ResultRowView[] mResultRowViews = new ResultRowView[3]; // Positive & Negative & Time elapsed

    private Toolbar toolBar;
    private String mLastBgHandledText;

    private Interpreter tflite;
    private Map<String, Integer> dic;
    private static final int MAX_SEQ_LEN = 40;
    private static final boolean DO_LOWER_CASE = false;
    private static final boolean PAD_TO_MAX_LENGTH = true;
    private FeatureConverter featureConverter;

    public void loadDictionaryFile() throws IOException {
        final String vocabFilePath = new File(
                Utils.assetFilePath(this, DIC_PATH)).getAbsolutePath();
        try (BufferedReader reader = new BufferedReader(new FileReader(new File(vocabFilePath)))) {
            int index = 0;
            while (reader.ready()) {
                String key = reader.readLine();
                dic.put(key, index++);
            }
        }
    }

    public void loadDictionary() {
        try {
            loadDictionaryFile();
            Log.v(TAG, "Dictionary loaded.");
        } catch (IOException ex) {
            Log.e(TAG, ex.getMessage());
        }
    }

    public MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
        try (AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
             FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    public void loadModel() {
        try {
            ByteBuffer buffer = loadModelFile(this.getAssets());
            Interpreter.Options opt = new Interpreter.Options();
            opt.setNumThreads(NUM_LITE_THREADS);
            tflite = new Interpreter(buffer, opt);
            Log.v(TAG, "TFLite model loaded.");
        } catch (IOException ex) {
            Log.e(TAG, ex.getMessage());
        }
    }

    private static class AnalysisResult {
        private final float[] scores;
        private final String[] className;
        private final long moduleForwardDuration;

        public AnalysisResult(float[] scores, long moduleForwardDuration) {
            this.scores = scores;
            this.moduleForwardDuration = moduleForwardDuration;
            this.className = new String[2];
            this.className[0] = "Negative";
            this.className[1] = "Positive";
        }
    }


    private Runnable mOnEditTextStopRunnable = () -> {
        final String text = mEditText.getText().toString();
        mBackgroundHandler.post(() -> {
            if (TextUtils.equals(text, mLastBgHandledText)) {
                return;
            }

            if (TextUtils.isEmpty(text)) {
                runOnUiThread(() -> applyUIEmptyTextState());
                mLastBgHandledText = null;
                return;
            }

            final AnalysisResult result = analyzeText(text);
            if (result != null) {
                runOnUiThread(() -> applyUIAnalysisResult(result));
                mLastBgHandledText = text;
            }
        });
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_nsmc);
        mEditText = findViewById(R.id.nsmc_edit_text);
        findViewById(R.id.nsmc_clear_button).setOnClickListener(v -> mEditText.setText(""));

        toolBar = findViewById(R.id.toolbar);
        toolBar.setTitle(R.string.nsmc_tensorflow);

        final ResultRowView headerRow = findViewById(R.id.nsmc_result_header_row);
        headerRow.nameTextView.setText(R.string.nsmc_sentiment);
        headerRow.scoreTextView.setText(R.string.nsmc_score);
        headerRow.setVisibility(View.VISIBLE);

        mResultRowViews[0] = findViewById(R.id.nsmc_top1_result_row);
        mResultRowViews[1] = findViewById(R.id.nsmc_top2_result_row);
        mResultRowViews[2] = findViewById(R.id.nsmc_time_row);
        mResultContent = findViewById(R.id.nsmc_result_content);

        mEditText.addTextChangedListener(new InternalTextWatcher());
    }

    @WorkerThread
    @Nullable
    private AnalysisResult analyzeText(final String text) {
        if (tflite == null) {
            loadModel();
        }
        if (dic == null) {
            dic = new HashMap<>();
            this.loadDictionary();
            featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_SEQ_LEN, PAD_TO_MAX_LENGTH);
        }

        Feature feature = featureConverter.convert(text);
        int curSeqLen = feature.inputIds.length;
        int[][] inputIds = new int[1][curSeqLen];

        for (int j = 0; j < curSeqLen; j++) {
            inputIds[0][j] = feature.inputIds[j];
        }

        Map<Integer, Object> output = new HashMap<>();
        float[][] softmaxLogits = new float[1][2];
        float[][] logits = new float[1][2];
        output.put(0, softmaxLogits);
        output.put(1, logits);

        final long moduleForwardStartTime = SystemClock.elapsedRealtime();
        tflite.runForMultipleInputsOutputs(new Object[]{inputIds}, output);
        final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

        float[] scores = new float[2];
        scores[0] = softmaxLogits[0][0];
        scores[1] = softmaxLogits[0][1];

        return new AnalysisResult(scores, moduleForwardDuration);
    }

    private void applyUIAnalysisResult(AnalysisResult result) {
        int first_idx, second_idx;
        if (result.scores[0] >= result.scores[1]) {
            first_idx = 0;
            second_idx = 1;
        } else {
            first_idx = 1;
            second_idx = 0;
        }
        setUiResultRowView(
                mResultRowViews[0],
                result.className[first_idx],
                String.format(Locale.US, SCORES_FORMAT, result.scores[first_idx]));
        setUiResultRowView(
                mResultRowViews[1],
                result.className[second_idx],
                String.format(Locale.US, SCORES_FORMAT, result.scores[second_idx]));
        setUiResultRowView(
                mResultRowViews[2],
                "Time elapsed",
                String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration)
        );
        mResultContent.setVisibility(View.VISIBLE);
    }

    private void applyUIEmptyTextState() {
        mResultContent.setVisibility(View.GONE);
    }

    private void setUiResultRowView(ResultRowView resultRowView, String name, String score) {
        resultRowView.nameTextView.setText(name);
        resultRowView.scoreTextView.setText(score);
        resultRowView.setProgressState(false);
    }

    @Override
    protected int getInfoViewCode() {
        return InfoViewFactory.INFO_VIEW_TYPE_TEXT_CLASSIFICATION;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
        if (dic != null) {
            dic.clear();
        }
    }

    private class InternalTextWatcher implements TextWatcher {
        @Override
        public void beforeTextChanged(CharSequence s, int start, int count, int after) {
        }

        @Override
        public void onTextChanged(CharSequence s, int start, int before, int count) {
        }

        @Override
        public void afterTextChanged(Editable s) {
            mUIHandler.removeCallbacks(mOnEditTextStopRunnable);
            mUIHandler.postDelayed(mOnEditTextStopRunnable, EDIT_TEXT_STOP_DELAY);
        }
    }

}
