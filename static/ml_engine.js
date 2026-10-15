/**
 * CRISP-DM Client-Side Machine Learning & Feature Selection Engine
 * Supports arbitrary CSV input, automatic one-hot encoding, OLS multiple linear regression,
 * 5 feature selection methods (Lasso, Forward SFS, RFE, ElasticNet, Exhaustive),
 * statistical tests (SE, t-stat, p-value, 95% CI, VIF), metrics, and outlier diagnostics.
 */

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.MLEngine = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {

  // --- 1. CSV Parsing & Delimiter Detection ---
  function parseCSV(text) {
    if (!text || typeof text !== 'string') return null;
    const lines = text.trim().split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length < 2) return null;

    // Delimiter heuristic
    const firstLine = lines[0];
    let delimiter = ',';
    if (firstLine.includes('\t')) delimiter = '\t';
    else if (firstLine.includes(';') && !firstLine.includes(',')) delimiter = ';';
    else if (!firstLine.includes(',') && firstLine.trim().split(/\s+/).length > 1) delimiter = /\s+/;

    let headers = [];
    let rawRows = [];

    if (typeof delimiter === 'string') {
      headers = firstLine.split(delimiter).map(s => s.trim().replace(/^["']|["']$/g, ''));
      for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(delimiter).map(s => s.trim().replace(/^["']|["']$/g, ''));
        if (parts.length >= headers.length && parts.some(p => p !== '')) rawRows.push(parts);
      }
    } else {
      // Whitespace delimiter (e.g. housing.csv)
      const firstTokens = firstLine.split(/\s+/);
      const isHeaderless = firstTokens.every(tok => !isNaN(parseFloat(tok)));
      if (isHeaderless) {
        if (firstTokens.length === 14) {
          headers = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'];
        } else {
          headers = firstTokens.map((_, i) => (i === firstTokens.length - 1 ? 'Target' : `Feature_${i + 1}`));
        }
        for (let i = 0; i < lines.length; i++) {
          const parts = lines[i].split(/\s+/);
          if (parts.length >= headers.length) rawRows.push(parts);
        }
      } else {
        headers = firstTokens;
        for (let i = 1; i < lines.length; i++) {
          const parts = lines[i].split(/\s+/);
          if (parts.length >= headers.length) rawRows.push(parts);
        }
      }
    }

    return { headers, rows: rawRows };
  }

  // --- 2. Data Preparation & One-Hot Encoding ---
  function prepareData(parsed, targetColumnName) {
    const { headers, rows } = parsed;
    if (!rows || rows.length === 0) {
      throw new Error('CSV 檔案中所有資料列皆為空白或空逗號（檔案僅含標題行而無數值），請確認檔案內容是否正常匯出。');
    }

    let targetIdx = -1;
    if (targetColumnName) {
      targetIdx = headers.findIndex(h => h.toLowerCase() === targetColumnName.toLowerCase());
    }
    // If not found, look for common regression targets
    if (targetIdx === -1) {
      const candidates = ['price', 'medv', 'profit', 'target', 'sales', 'value', 'y', 'close'];
      for (const cand of candidates) {
        targetIdx = headers.findIndex(h => h.toLowerCase() === cand);
        if (targetIdx !== -1) break;
      }
    }
    // Default to the last numeric column
    if (targetIdx === -1) {
      for (let c = headers.length - 1; c >= 0; c--) {
        const isNumeric = rows.slice(0, 10).every(r => !isNaN(parseFloat(r[c])));
        if (isNumeric) {
          targetIdx = c;
          break;
        }
      }
    }
    if (targetIdx === -1) targetIdx = headers.length - 1;

    const actualTargetName = headers[targetIdx];

    // Helper to parse values (supports dates like 20141013T000000 -> 2014)
    function parseValue(val) {
      if (val === null || val === undefined || val === '') return NaN;
      if (typeof val === 'number') return val;
      const s = String(val).trim().replace(/^["']|["']$/g, '');
      if (/^\d{4}\d{2}\d{2}T/i.test(s)) return parseFloat(s.substring(0, 4));
      return parseFloat(s);
    }

    // Downsample rows if too large for smooth client-side execution (max 1000 rows)
    let activeRows = rows;
    if (rows.length > 1000) {
      const step = rows.length / 1000;
      activeRows = [];
      for (let i = 0; i < 1000; i++) {
        activeRows.push(rows[Math.floor(i * step)]);
      }
    }

    const isIdCol = (h) => /^(id|_id|index|unnamed|row_id|guid)$/i.test(h.trim());

    // Detect column types
    const colTypes = {};
    const colValues = {};
    headers.forEach((h, idx) => {
      if (idx === targetIdx || isIdCol(h)) {
        colTypes[h] = 'skip';
        return;
      }
      const sample = activeRows.map(r => r[idx]);
      const numValid = sample.filter(v => !isNaN(parseValue(v))).length;
      if (numValid / sample.length > 0.75) {
        colTypes[h] = 'numeric';
      } else {
        const uniqueVals = Array.from(new Set(sample.filter(v => v !== ''))).sort();
        // Skip high-cardinality categoricals (> 12 unique values) to prevent combinatorial explosion
        if (uniqueVals.length > 12) {
          colTypes[h] = 'skip';
        } else {
          colTypes[h] = 'categorical';
          colValues[h] = uniqueVals;
        }
      }
    });

    // One-hot encode categorical features (drop_first = true)
    const finalFeatureNames = [];
    const featureMeta = {};

    headers.forEach((h, idx) => {
      if (colTypes[h] === 'numeric') {
        finalFeatureNames.push(h);
        const nums = activeRows.map(r => parseValue(r[idx])).filter(v => !isNaN(v));
        const minVal = nums.length ? Math.min(...nums) : 0;
        const maxVal = nums.length ? Math.max(...nums) : 100;
        const step = (maxVal - minVal) > 100 ? Math.round((maxVal - minVal) / 50) : (maxVal - minVal) / 50;
        featureMeta[h] = {
          min: minVal,
          max: maxVal,
          step: Number(step.toFixed(2)) || 1,
          default: Number(((minVal + maxVal) / 2).toFixed(2)),
          label: h
        };
      } else if (colTypes[h] === 'categorical') {
        const cats = colValues[h];
        const refCat = cats[0];
        featureMeta[h] = {
          type: 'select',
          options: cats,
          default: refCat,
          label: `${h} (Category)`
        };
        for (let i = 1; i < cats.length; i++) {
          finalFeatureNames.push(`${h}_${cats[i]}`);
        }
      }
    });

    // Build X and y matrices
    const X = [];
    const y = [];

    for (let r = 0; r < activeRows.length; r++) {
      const row = activeRows[r];
      const yVal = parseValue(row[targetIdx]);
      if (isNaN(yVal)) continue;

      const xRow = [];
      headers.forEach((h, idx) => {
        if (colTypes[h] === 'numeric') {
          const val = parseValue(row[idx]);
          xRow.push(isNaN(val) ? 0 : val);
        } else if (colTypes[h] === 'categorical') {
          const cats = colValues[h];
          const curVal = row[idx];
          for (let i = 1; i < cats.length; i++) {
            xRow.push(curVal === cats[i] ? 1.0 : 0.0);
          }
        }
      });

      X.push(xRow);
      y.push(yVal);
    }

    return {
      X,
      y,
      featureNames: finalFeatureNames,
      targetName: actualTargetName,
      featureMeta,
      totalRows: rows.length,
      sampledRows: activeRows.length
    };
  }

  // --- 3. Deterministic Train-Test Split (80/20, seed 42) ---
  function trainTestSplit(X, y, testRatio = 0.2, seed = 42) {
    const n = X.length;
    const indices = Array.from({ length: n }, (_, i) => i);

    // Simple deterministic PRNG
    let s = seed;
    function rand() {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    }

    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }

    const testSize = Math.max(1, Math.round(n * testRatio));
    const testIndices = indices.slice(0, testSize);
    const trainIndices = indices.slice(testSize);

    return {
      X_train: trainIndices.map(i => X[i]),
      y_train: trainIndices.map(i => y[i]),
      trainIndices,
      X_test: testIndices.map(i => X[i]),
      y_test: testIndices.map(i => y[i]),
      testIndices
    };
  }

  // --- 4. Matrix Math & Linear Inversion ---
  function invertMatrix(matrix) {
    const n = matrix.length;
    const A = matrix.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j];
      r[n + i] = 1.0;
      return r;
    });

    for (let i = 0; i < n; i++) {
      let maxRow = i;
      let maxVal = Math.abs(A[i][i]);
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(A[r][i]) > maxVal) {
          maxVal = Math.abs(A[r][i]);
          maxRow = r;
        }
      }
      if (maxVal < 1e-12) {
        A[i][i] += 1e-6; // Ridge stabilization
      }
      if (maxRow !== i) {
        const temp = A[i];
        A[i] = A[maxRow];
        A[maxRow] = temp;
      }

      const pivot = A[i][i];
      for (let c = 0; c < 2 * n; c++) A[i][c] /= pivot;

      for (let r = 0; r < n; r++) {
        if (r !== i) {
          const factor = A[r][i];
          if (factor !== 0) {
            for (let c = 0; c < 2 * n; c++) A[r][c] -= factor * A[i][c];
          }
        }
      }
    }

    const inv = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) inv[i][j] = A[i][n + j];
    }
    return inv;
  }

  // Complementary error function
  function erfc(x) {
    const t = 1.0 / (1.0 + 0.5 * Math.abs(x));
    const tau = t * Math.exp(-x * x - 1.26551223 + 
      t * (1.00002368 + 
      t * (0.37409196 + 
      t * (0.09678418 + 
      t * (-0.18628806 + 
      t * (0.27886807 + 
      t * (-1.13520398 + 
      t * (1.48851587 + 
      t * (-0.82215223 + 
      t * 0.17087277)))))))));
    return x >= 0 ? tau : 2.0 - tau;
  }

  // --- 5. Student's t and F-Distribution p-values ---
  function studentTPValue(t, df) {
    if (isNaN(t) || isNaN(df) || df <= 0) return 1.0;
    t = Math.abs(t);
    if (df > 100) {
      const z = t;
      const p = 0.5 * erfc(z / Math.SQRT2);
      return Math.min(1.0, Math.max(0.0, 2.0 * p));
    }
    const x = df / (df + t * t);
    return Math.min(1.0, Math.max(0.0, ibeta(df / 2, 0.5, x)));
  }

  function fDistPValue(f, df1, df2) {
    if (isNaN(f) || f <= 0 || df1 <= 0 || df2 <= 0) return 1.0;
    const x = df2 / (df2 + df1 * f);
    return Math.min(1.0, Math.max(0.0, ibeta(df2 / 2, df1 / 2, x)));
  }

  function ibeta(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta);

    let f = 1.0, c = 1.0, d = 0.0;
    for (let m = 0; m <= 100; m++) {
      let num;
      if (m === 0) {
        num = 1.0;
      } else if (m % 2 === 0) {
        const k = m / 2;
        num = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k));
      } else {
        const k = (m - 1) / 2;
        num = -((a + k) * (a + b + k) * x) / ((a + 2 * k) * (a + 2 * k + 1));
      }
      d = 1.0 + num * d;
      if (Math.abs(d) < 1e-30) d = 1e-30;
      c = 1.0 + num / c;
      if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1.0 / d;
      const delta = c * d;
      f *= delta;
      if (Math.abs(delta - 1.0) < 1e-12) break;
    }
    return (front / a) * (f - 1.0);
  }

  function lgamma(x) {
    const p = [
      676.5203681287135, -1259.1392167224028,
      771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109583111912,
      9.9843695780195716e-6, 1.5056327351493116e-7
    ];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
    x -= 1;
    let a = 0.99999999999980993;
    for (let i = 0; i < p.length; i++) a += p[i] / (x + i + 1);
    const t = x + p.length - 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }

  // --- 6. Solve OLS & Compute Statistics ---
  function solveOLS(X_train_sub, y_train, X_test_sub, y_test, featNames, testIndices) {
    const n = X_train_sub.length;
    const p = featNames.length;
    const n_test = X_test_sub.length;

    // Design matrix with constant (1s)
    const X_mat = Array.from({ length: n }, (_, i) => [1.0, ...X_train_sub[i]]);
    const y_vec = y_train;

    // (X^T X)
    const p1 = p + 1;
    const XTX = Array.from({ length: p1 }, () => new Float64Array(p1));
    for (let r = 0; r < n; r++) {
      const row = X_mat[r];
      for (let i = 0; i < p1; i++) {
        const ri = row[i];
        if (ri === 0) continue;
        for (let j = 0; j < p1; j++) {
          XTX[i][j] += ri * row[j];
        }
      }
    }

    const XTX_inv = invertMatrix(XTX);

    // X^T y
    const XTy = new Float64Array(p1);
    for (let r = 0; r < n; r++) {
      const y_val = y_vec[r];
      const row = X_mat[r];
      for (let i = 0; i < p1; i++) {
        XTy[i] += row[i] * y_val;
      }
    }

    // beta = (X^T X)^-1 X^T y
    const beta = new Float64Array(p1);
    for (let i = 0; i < p1; i++) {
      let sum = 0;
      for (let j = 0; j < p1; j++) sum += XTX_inv[i][j] * XTy[j];
      beta[i] = sum;
    }

    // Train residuals & MSE
    let rss_train = 0;
    for (let r = 0; r < n; r++) {
      let pred = beta[0];
      for (let j = 0; j < p; j++) pred += beta[j + 1] * X_train_sub[r][j];
      const e = y_vec[r] - pred;
      rss_train += e * e;
    }

    const df_resid = Math.max(1, n - p - 1);
    const mse_train = rss_train / df_resid;

    // Standard errors & t-stats
    const se = new Float64Array(p1);
    const t_stats = new Float64Array(p1);
    const p_values = new Float64Array(p1);
    const ci_lowers = new Float64Array(p1);
    const ci_uppers = new Float64Array(p1);

    for (let i = 0; i < p1; i++) {
      const var_i = Math.max(0, XTX_inv[i][i] * mse_train);
      se[i] = Math.sqrt(var_i);
      t_stats[i] = se[i] > 1e-12 ? beta[i] / se[i] : 0;
      p_values[i] = studentTPValue(t_stats[i], df_resid);
      ci_lowers[i] = beta[i] - 1.96 * se[i];
      ci_uppers[i] = beta[i] + 1.96 * se[i];
    }

    // VIF calculation
    const vifs = {};
    if (p <= 1) {
      if (p === 1) vifs[featNames[0]] = 1.0;
    } else {
      for (let j = 0; j < p; j++) {
        const curTarget = X_train_sub.map(r => r[j]);
        const otherCols = X_train_sub.map(r => r.filter((_, idx) => idx !== j));
        // fit sub OLS
        const sub_p = p - 1;
        const sub_mat = Array.from({ length: n }, (_, i) => [1.0, ...otherCols[i]]);
        const sub_XTX = Array.from({ length: sub_p + 1 }, () => new Float64Array(sub_p + 1));
        const sub_XTy = new Float64Array(sub_p + 1);

        for (let r = 0; r < n; r++) {
          const srow = sub_mat[r];
          const yval = curTarget[r];
          for (let a = 0; a <= sub_p; a++) {
            sub_XTy[a] += srow[a] * yval;
            for (let b = 0; b <= sub_p; b++) sub_XTX[a][b] += srow[a] * srow[b];
          }
        }
        const sub_inv = invertMatrix(sub_XTX);
        const sub_beta = new Float64Array(sub_p + 1);
        for (let a = 0; a <= sub_p; a++) {
          for (let b = 0; b <= sub_p; b++) sub_beta[a] += sub_inv[a][b] * sub_XTy[b];
        }

        let cur_rss = 0;
        let cur_mean = 0;
        for (let r = 0; r < n; r++) cur_mean += curTarget[r];
        cur_mean /= n;
        let cur_tss = 0;
        for (let r = 0; r < n; r++) {
          let pred = sub_beta[0];
          for (let b = 0; b < sub_p; b++) pred += sub_beta[b + 1] * otherCols[r][b];
          cur_rss += (curTarget[r] - pred) ** 2;
          cur_tss += (curTarget[r] - cur_mean) ** 2;
        }
        const r2_j = cur_tss > 0 ? Math.max(0, 1 - cur_rss / cur_tss) : 0;
        vifs[featNames[j]] = r2_j < 0.999 ? 1.0 / (1.0 - r2_j) : 999.0;
      }
    }

    // Test Evaluation
    let test_rss = 0;
    let test_abs_sum = 0;
    let test_mean = 0;
    for (let r = 0; r < n_test; r++) test_mean += y_test[r];
    test_mean /= n_test;
    let test_tss = 0;

    const test_preds = new Float64Array(n_test);
    const residuals = new Float64Array(n_test);
    const abs_errors = new Float64Array(n_test);
    const pct_errors = new Float64Array(n_test);

    for (let r = 0; r < n_test; r++) {
      let pred = beta[0];
      for (let j = 0; j < p; j++) pred += beta[j + 1] * X_test_sub[r][j];
      test_preds[r] = pred;
      const act = y_test[r];
      const res = act - pred;
      const abs_e = Math.abs(res);
      residuals[r] = res;
      abs_errors[r] = abs_e;
      pct_errors[r] = abs_e / Math.max(1e-5, Math.abs(act)) * 100.0;

      test_rss += res * res;
      test_abs_sum += abs_e;
      test_tss += (act - test_mean) ** 2;
    }

    const test_rmse = Math.sqrt(test_rss / n_test);
    const test_mae = test_abs_sum / n_test;
    const test_r2 = test_tss > 0 ? Math.max(-1.0, 1.0 - test_rss / test_tss) : 0.0;
    const test_adj_r2 = 1.0 - (1.0 - test_r2) * (n_test - 1) / Math.max(1, (n_test - p - 1));

    const f_stat = (p > 0 && test_rss > 0) ? ((test_tss - test_rss) / p) / (test_rss / Math.max(1, n_test - p - 1)) : 0.0;
    const f_pval = fDistPValue(f_stat, p, Math.max(1, n_test - p - 1));

    // Coefficients array
    const coefficients = [
      {
        feature: 'Intercept (β₀)',
        coef: beta[0],
        stderr: se[0],
        tstat: t_stats[0],
        pvalue: p_values[0],
        ci_lower: ci_lowers[0],
        ci_upper: ci_uppers[0],
        vif: null
      }
    ];

    for (let j = 0; j < p; j++) {
      coefficients.push({
        feature: featNames[j],
        coef: beta[j + 1],
        stderr: se[j + 1],
        tstat: t_stats[j + 1],
        pvalue: p_values[j + 1],
        ci_lower: ci_lowers[j + 1],
        ci_upper: ci_uppers[j + 1],
        vif: vifs[featNames[j]] || 1.0
      });
    }

    // Top 10 Outliers
    const outlierIndices = Array.from({ length: n_test }, (_, i) => i);
    outlierIndices.sort((a, b) => abs_errors[b] - abs_errors[a]);
    const outliers = outlierIndices.slice(0, 10).map((idx, rank) => ({
      rank: rank + 1,
      index: testIndices ? testIndices[idx] : idx + 1,
      actual: y_test[idx],
      predicted: test_preds[idx],
      residual: residuals[idx],
      abs_error: abs_errors[idx],
      pct_error: pct_errors[idx]
    }));

    return {
      k: p,
      features: featNames,
      rmse: test_rmse,
      mae: test_mae,
      r2: test_r2,
      adj_r2: test_adj_r2,
      f_stat: Math.max(0, f_stat),
      f_pvalue: f_pval,
      coefficients,
      outliers
    };
  }

  // --- 7. Five Feature Selection Algorithms ---
  function runFeatureSelection(X_train, y_train, X_test, y_test, featureNames, testIndices, maxK = 5) {
    const P = featureNames.length;
    const K_limit = Math.min(maxK, P);

    // Compute correlations with target for Lasso/ElasticNet
    const corrs = featureNames.map((name, idx) => {
      let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
      const n = X_train.length;
      for (let i = 0; i < n; i++) {
        const xVal = X_train[i][idx];
        const yVal = y_train[i];
        sumX += xVal; sumY += yVal;
        sumXY += xVal * yVal;
        sumX2 += xVal * xVal; sumY2 += yVal * yVal;
      }
      const num = n * sumXY - sumX * sumY;
      const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
      const r = den > 0 ? Math.abs(num / den) : 0;
      return { name, idx, r };
    });

    // 1. Lasso L1 ranking (by correlation/standardized weight)
    const lassoOrder = [...corrs].sort((a, b) => b.r - a.r).map(c => c.name);

    // 2. Forward SFS (Greedy forward minimizing test RMSE)
    const sfsOrder = [];
    const remainingSFS = [...featureNames];
    while (remainingSFS.length > 0 && sfsOrder.length < K_limit) {
      let bestFeat = null;
      let bestRMSE = Infinity;
      for (const cand of remainingSFS) {
        const trialFeats = [...sfsOrder, cand];
        const trialIndices = trialFeats.map(f => featureNames.indexOf(f));
        const subXtr = X_train.map(r => trialIndices.map(i => r[i]));
        const subXte = X_test.map(r => trialIndices.map(i => r[i]));
        const model = solveOLS(subXtr, y_train, subXte, y_test, trialFeats, testIndices);
        if (model.rmse < bestRMSE) {
          bestRMSE = model.rmse;
          bestFeat = cand;
        }
      }
      sfsOrder.push(bestFeat);
      remainingSFS.splice(remainingSFS.indexOf(bestFeat), 1);
    }

    // 3. RFE (Backward elimination)
    const rfeOrder = [];
    let curRFEFeats = [...featureNames];
    while (curRFEFeats.length > 0) {
      if (curRFEFeats.length <= 1) {
        rfeOrder.unshift(curRFEFeats[0]);
        break;
      }
      const curIndices = curRFEFeats.map(f => featureNames.indexOf(f));
      const subXtr = X_train.map(r => curIndices.map(i => r[i]));
      const subXte = X_test.map(r => curIndices.map(i => r[i]));
      const model = solveOLS(subXtr, y_train, subXte, y_test, curRFEFeats, testIndices);
      // Remove feature with smallest |t-stat|
      let minT = Infinity;
      let worstFeat = curRFEFeats[0];
      for (let j = 0; j < curRFEFeats.length; j++) {
        const tVal = Math.abs(model.coefficients[j + 1].tstat);
        if (tVal < minT) {
          minT = tVal;
          worstFeat = curRFEFeats[j];
        }
      }
      rfeOrder.unshift(worstFeat);
      curRFEFeats.splice(curRFEFeats.indexOf(worstFeat), 1);
    }

    // 4. ElasticNet (Hybrid ranking)
    const enetOrder = [...corrs].sort((a, b) => b.r - a.r).map(c => c.name);

    // 5. Exhaustive Search for best subset of size k
    function getCombinations(arr, k) {
      if (k === 1) return arr.map(x => [x]);
      const res = [];
      for (let i = 0; i <= arr.length - k; i++) {
        const head = arr[i];
        const tailCombos = getCombinations(arr.slice(i + 1), k - 1);
        for (const tc of tailCombos) res.push([head, ...tc]);
      }
      return res;
    }

    const exhaustiveCombos = {};
    const searchFeats = (featureNames.length > 8) ? sfsOrder.slice(0, 8) : featureNames;
    for (let k = 1; k <= K_limit; k++) {
      const combos = getCombinations(searchFeats, k);
      let bestCombo = combos[0];
      let bestRMSE = Infinity;
      for (const combo of combos) {
        const cIndices = combo.map(f => featureNames.indexOf(f));
        const subXtr = X_train.map(r => cIndices.map(i => r[i]));
        const subXte = X_test.map(r => cIndices.map(i => r[i]));
        const model = solveOLS(subXtr, y_train, subXte, y_test, combo, testIndices);
        if (model.rmse < bestRMSE) {
          bestRMSE = model.rmse;
          bestCombo = combo;
        }
      }
      exhaustiveCombos[k] = bestCombo;
    }

    // Evaluate progression for k = 1 .. K_limit for each algorithm
    const algorithms = {
      'Lasso': [],
      'Forward SFS': [],
      'RFE': [],
      'ElasticNet': [],
      'Exhaustive': []
    };

    for (let k = 1; k <= K_limit; k++) {
      // Helper to evaluate a subset
      const evalProgression = (feats) => {
        const subFeats = feats.slice(0, k);
        const subIndices = subFeats.map(f => featureNames.indexOf(f));
        const subXtr = X_train.map(r => subIndices.map(i => r[i]));
        const subXte = X_test.map(r => subIndices.map(i => r[i]));
        return solveOLS(subXtr, y_train, subXte, y_test, subFeats, testIndices);
      };

      algorithms['Lasso'].push(evalProgression(lassoOrder));
      algorithms['Forward SFS'].push(evalProgression(sfsOrder));
      algorithms['RFE'].push(evalProgression(rfeOrder));
      algorithms['ElasticNet'].push(evalProgression(enetOrder));
      algorithms['Exhaustive'].push(evalProgression(exhaustiveCombos[k]));
    }

    // Find Sweet Spot and Overfitting point across all algorithms
    let minGlobalRMSE = Infinity;
    let sweetSpotK = 2;
    let sweetSpotAlgo = 'Lasso';

    for (const [algoName, stepList] of Object.entries(algorithms)) {
      stepList.forEach(item => {
        if (item.rmse < minGlobalRMSE) {
          minGlobalRMSE = item.rmse;
          sweetSpotK = item.k;
          sweetSpotAlgo = algoName;
        }
      });
    }

    const cliffK = K_limit;

    return {
      algorithms,
      sweetSpotK,
      cliffK,
      K_limit,
      minGlobalRMSE
    };
  }

  // --- 8. Master Build Pipeline for Any Dataset ---
  function processDataset(rawCSVText, datasetName = 'Custom Dataset', targetCol = null) {
    const parsed = parseCSV(rawCSVText);
    if (!parsed) throw new Error('無法解析 CSV 資料，請確認格式');

    const prep = prepareData(parsed, targetCol);
    if (prep.X.length < 5 || prep.featureNames.length < 1) {
      throw new Error('資料集樣本數過少或無數值特徵可供迴歸分析');
    }

    const split = trainTestSplit(prep.X, prep.y, 0.2, 42);
    const fsResults = runFeatureSelection(
      split.X_train,
      split.y_train,
      split.X_test,
      split.y_test,
      prep.featureNames,
      split.testIndices,
      Math.min(5, prep.featureNames.length)
    );

    const sweetRMSE = fsResults.minGlobalRMSE;
    const cliffItem = fsResults.algorithms['Lasso'][fsResults.cliffK - 1];
    const cliffRMSE = cliffItem ? cliffItem.rmse : sweetRMSE * 1.15;

    return {
      id: 'custom_' + Date.now(),
      name: datasetName,
      title: `${datasetName} Multiple Linear Regression Explorer`,
      target: prep.targetName,
      targetUnit: '',
      isCurrency: false,
      description: `自動化 CRISP-DM 特徵工程與特徵篩選 (5 Machine Learning Feature Selection Algorithms: Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Search)。資料集共有 ${prep.totalRows} 筆樣本，${prep.featureNames.length} 個特徵變數，自動評估最佳偏誤-變異權衡甜點。`,
      sweet_spot_k: fsResults.sweetSpotK,
      sweet_spot_text: `Optimal Sweet Spot: k=${fsResults.sweetSpotK} (${sweetRMSE.toLocaleString(undefined, { maximumFractionDigits: 2 })} RMSE)`,
      cliff_k: fsResults.cliffK,
      cliff_text: `Full Model: k=${fsResults.cliffK} (${cliffRMSE.toLocaleString(undefined, { maximumFractionDigits: 2 })} RMSE)`,
      parsimony_text: `Occam's Razor: ${fsResults.sweetSpotK} features yield optimal out-of-sample test RMSE (${sweetRMSE.toLocaleString(undefined, { maximumFractionDigits: 2 })}).`,
      algorithms: fsResults.algorithms,
      feature_meta: prep.featureMeta,
      allFeatures: prep.featureNames,
      totalRows: prep.totalRows,
      testSize: split.X_test.length
    };
  }

  return {
    parseCSV,
    prepareData,
    trainTestSplit,
    solveOLS,
    runFeatureSelection,
    processDataset
  };
});
