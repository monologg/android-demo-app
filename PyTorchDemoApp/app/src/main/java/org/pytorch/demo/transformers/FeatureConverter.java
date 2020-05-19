/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.pytorch.demo.transformers;

import org.pytorch.demo.tokenization.FullTokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Convert String to features that can be fed into BERT model.
 */
public final class FeatureConverter {
    private final FullTokenizer tokenizer;
    private final int maxSeqLen;

    public FeatureConverter(
            Map<String, Integer> inputDic, boolean doLowerCase, int maxSeqLen) {
        this.tokenizer = new FullTokenizer(inputDic, doLowerCase);
        this.maxSeqLen = maxSeqLen;
    }

    public Feature convert(String text) {
        List<String> tokens = tokenizer.tokenize(text);
        tokens.add(0, "[CLS]"); // Start of generating the features.
        if (tokens.size() > maxSeqLen - 1) {
            tokens = tokens.subList(0, maxSeqLen - 1);
        }
        tokens.add("[SEP]"); // For Separation.

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);

        return new Feature(inputIds);
    }
}
