package org.pytorch.demo.nlp;

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

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.BaseModuleActivity;
import org.pytorch.demo.InfoViewFactory;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.transformers.Feature;
import org.pytorch.demo.transformers.FeatureConverter;
import org.pytorch.demo.view.ResultRowView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class IMDBPytorchActivity extends BaseModuleActivity {
    private static final String TAG = "IMDBPytorchDemo";
    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
    private static final String MODEL_PATH = "imdb_small.pt";
    private static final String DIC_PATH = "imdb_vocab.txt";

    private static final long EDIT_TEXT_STOP_DELAY = 600l;
    private static final String FORMAT_MS = "%dms";
    private static final String SCORES_FORMAT = "%.2f";

    private EditText mEditText;
    private View mResultContent;
    private ResultRowView[] mResultRowViews = new ResultRowView[3]; // Positive & Negative & Time elapsed

    private Module mModule;
    private String mModuleAssetName;

    private Toolbar toolBar;
    private String mLastBgHandledText;

    private Map<String, Integer> dic;
    private static final int MAX_SEQ_LEN = 20;
    private static final boolean DO_LOWER_CASE = true;
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
        setContentView(R.layout.activity_imdb);
        mEditText = findViewById(R.id.imdb_edit_text);
        findViewById(R.id.imdb_clear_button).setOnClickListener(v -> mEditText.setText(""));

        toolBar = findViewById(R.id.toolbar);
        toolBar.setTitle(R.string.imdb_pytorch);

        final ResultRowView headerRow = findViewById(R.id.imdb_result_header_row);
        headerRow.nameTextView.setText(R.string.imdb_sentiment);
        headerRow.scoreTextView.setText(R.string.imdb_score);
        headerRow.setVisibility(View.VISIBLE);

        mResultRowViews[0] = findViewById(R.id.imdb_top1_result_row);
        mResultRowViews[1] = findViewById(R.id.imdb_top2_result_row);
        mResultRowViews[2] = findViewById(R.id.imdb_time_row);
        mResultContent = findViewById(R.id.imdb_result_content);

        mEditText.addTextChangedListener(new InternalTextWatcher());
    }

    protected String getModuleAssetName() {
        if (!TextUtils.isEmpty(mModuleAssetName)) {
            return mModuleAssetName;
        }
        final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
                ? moduleAssetNameFromIntent
                : MODEL_PATH;

        return mModuleAssetName;
    }

    @WorkerThread
    @Nullable
    private AnalysisResult analyzeText(final String text) {
        if (mModule == null) {
            final String moduleFileAbsoluteFilePath = new File(
                    Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
            mModule = Module.load(moduleFileAbsoluteFilePath);
        }
        if (dic == null) {
            dic = new HashMap<>();
            this.loadDictionary();
            featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_SEQ_LEN);
        }

        Feature feature = featureConverter.convert(text);

        int curSeqLen = feature.inputIds.length;

        long[] inputIds = new long[curSeqLen];

        for (int j = 0; j < curSeqLen; j++) {
            inputIds[j] = feature.inputIds[j];
        }

        final long[] shape = new long[]{1, curSeqLen};

        final Tensor inputIdsTensor = Tensor.fromBlob(inputIds, shape);

        final long moduleForwardStartTime = SystemClock.elapsedRealtime();
        final IValue output = mModule.forward(IValue.from(inputIdsTensor));
        final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;
        IValue[] outputTuple = output.toTuple();
        final Tensor outputTensor = outputTuple[0].toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();

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
        if (mModule != null) {
            mModule.destroy();
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
