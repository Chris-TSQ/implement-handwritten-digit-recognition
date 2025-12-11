import React, { useState, useRef, useEffect } from "react";

// Neural Network Class
class neuralNetwork {
  constructor(inputnodes, hiddennodes, outputnodes, learningrate) {
    this.inodes = inputnodes;
    this.hnodes = hiddennodes;
    this.onodes = outputnodes;

    this.wih = this.randomNormal(
      0.0,
      Math.pow(this.inodes, -0.5),
      this.hnodes,
      this.inodes
    );
    this.who = this.randomNormal(
      0.0,
      Math.pow(this.hnodes, -0.5),
      this.onodes,
      this.hnodes
    );

    this.lr = learningrate;
    this.activation_function = (x) => 1.0 / (1.0 + Math.exp(-x));
  }

  randomNormal(mean, stddev, rows, cols) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      const row = [];
      for (let j = 0; j < cols; j++) {
        const u1 = Math.random();
        const u2 = Math.random();
        const randStdNormal =
          Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
        row.push(mean + stddev * randStdNormal);
      }
      matrix.push(row);
    }
    return matrix;
  }

  dot(a, b) {
    const result = [];
    for (let i = 0; i < a.length; i++) {
      let sum = 0;
      for (let j = 0; j < a[i].length; j++) {
        sum += a[i][j] * b[j];
      }
      result.push(sum);
    }
    return result;
  }

  transpose(matrix) {
    if (typeof matrix[0] === "number") {
      return matrix.map((val) => [val]);
    }
    const rows = matrix.length;
    const cols = matrix[0].length;
    const result = [];
    for (let j = 0; j < cols; j++) {
      const row = [];
      for (let i = 0; i < rows; i++) {
        row.push(matrix[i][j]);
      }
      result.push(row);
    }
    return result;
  }

  train(inputs_list, targets_list) {
    const inputs = inputs_list;
    const targets = targets_list;

    const hidden_inputs = this.dot(this.wih, inputs);
    const hidden_outputs = hidden_inputs.map((x) =>
      this.activation_function(x)
    );

    const final_inputs = this.dot(this.who, hidden_outputs);
    const final_outputs = final_inputs.map((x) => this.activation_function(x));

    const output_errors = targets.map((t, i) => t - final_outputs[i]);
    const who_T = this.transpose(this.who);
    const hidden_errors = this.dot(who_T, output_errors);

    for (let i = 0; i < this.onodes; i++) {
      for (let j = 0; j < this.hnodes; j++) {
        this.who[i][j] +=
          this.lr *
          output_errors[i] *
          final_outputs[i] *
          (1.0 - final_outputs[i]) *
          hidden_outputs[j];
      }
    }

    for (let i = 0; i < this.hnodes; i++) {
      for (let j = 0; j < this.inodes; j++) {
        this.wih[i][j] +=
          this.lr *
          hidden_errors[i] *
          hidden_outputs[i] *
          (1.0 - hidden_outputs[i]) *
          inputs[j];
      }
    }
  }

  query(inputs_list) {
    const inputs = inputs_list;
    const hidden_inputs = this.dot(this.wih, inputs);
    const hidden_outputs = hidden_inputs.map((x) =>
      this.activation_function(x)
    );
    const final_inputs = this.dot(this.who, hidden_outputs);
    const final_outputs = final_inputs.map((x) => this.activation_function(x));
    return final_outputs;
  }
}

function MNISTDigitRecognition() {
  const [n, setN] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [testAccuracy, setTestAccuracy] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [digitSegments, setDigitSegments] = useState([]);
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [trainingData, setTrainingData] = useState(null);
  const [testData, setTestData] = useState(null);

  useEffect(() => {
    const input_nodes = 784;
    const hidden_nodes = 200;
    const output_nodes = 10;
    const learning_rate = 0.1;

    const network = new neuralNetwork(
      input_nodes,
      hidden_nodes,
      output_nodes,
      learning_rate
    );
    setN(network);
  }, []);

  const loadMNISTData = async (file, isTrainingSet) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target.result;
        const lines = text.split("\n").filter((line) => line.trim().length > 0);

        const data = lines.map((line) => {
          const all_values = line.split(",").map((v) => parseFloat(v));
          const label = parseInt(all_values[0]);
          const inputs = all_values
            .slice(1)
            .map((v) => (v / 255.0) * 0.99 + 0.01);
          return { label, inputs };
        });

        if (isTrainingSet) {
          setTrainingData(data);
        } else {
          setTestData(data);
        }
        resolve(data);
      };
      reader.readAsText(file);
    });
  };

  const trainNetwork = async () => {
    if (!n || !trainingData) {
      alert("Please load training data first!");
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);

    const epochs = 5;
    const totalIterations = epochs * trainingData.length;
    let iteration = 0;

    for (let e = 0; e < epochs; e++) {
      for (let i = 0; i < trainingData.length; i++) {
        const record = trainingData[i];
        const targets = new Array(10).fill(0.01);
        targets[record.label] = 0.99;
        n.train(record.inputs, targets);

        iteration++;
        if (iteration % 100 === 0) {
          setTrainingProgress(Math.round((iteration / totalIterations) * 100));
          await new Promise((resolve) => setTimeout(resolve, 0));
        }
      }
    }

    setTrainingProgress(100);
    setIsTraining(false);
    alert("Training complete!");
  };

  const testNetwork = () => {
    if (!n || !testData) {
      alert("Please load test data and train the network first!");
      return;
    }

    let scorecard = [];

    for (let i = 0; i < testData.length; i++) {
      const record = testData[i];
      const correct_label = record.label;
      const outputs = n.query(record.inputs);
      const label = outputs.indexOf(Math.max(...outputs));

      if (label === correct_label) {
        scorecard.push(1);
      } else {
        scorecard.push(0);
      }
    }

    const scorecard_array_sum = scorecard.reduce((a, b) => a + b, 0);
    const performance = scorecard_array_sum / scorecard.length;
    const accuracy = (performance * 100).toFixed(2);
    setTestAccuracy(accuracy);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setConfidence(null);
    setDigitSegments([]);
  };

  useEffect(() => {
    if (canvasRef.current) {
      clearCanvas();
    }
  }, []);

  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const draw = (e) => {
    if (!isDrawing && e.type !== "mousedown" && e.type !== "touchstart") return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();

    let x, y;
    if (e.type.includes("touch")) {
      e.preventDefault();
      x = e.touches[0].clientX - rect.left;
      y = e.touches[0].clientY - rect.top;
    } else {
      x = e.clientX - rect.left;
      y = e.clientY - rect.top;
    }

    // Scale coordinates to canvas size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    x *= scaleX;
    y *= scaleY;

    // Draw with larger brush for better visibility
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, 20, 0, Math.PI * 2);
    ctx.fill();
  };

  const segmentDigits = (canvas) => {
    const ctx = canvas.getContext("2d");
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = imageData.data;
    const width = canvas.width;
    const height = canvas.height;

    const projection = new Array(width).fill(0);
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const idx = (y * width + x) * 4;
        if (pixels[idx] > 128) {
          projection[x]++;
        }
      }
    }

    const threshold = 5;
    const segments = [];
    let inDigit = false;
    let start = 0;

    for (let x = 0; x < width; x++) {
      if (!inDigit && projection[x] > threshold) {
        start = Math.max(0, x - 10);
        inDigit = true;
      } else if (inDigit && projection[x] <= threshold) {
        let isEnd = true;
        for (let look = 1; look < 15 && x + look < width; look++) {
          if (projection[x + look] > threshold) {
            isEnd = false;
            break;
          }
        }

        if (isEnd) {
          const end = Math.min(width, x + 10);
          if (end - start > 15) {
            segments.push({ start, end });
          }
          inDigit = false;
        }
      }
    }

    if (inDigit) {
      segments.push({ start, end: width });
    }

    return segments;
  };

  const preprocessDigitForMNIST = (sourceCanvas, segment) => {
    const ctx = sourceCanvas.getContext("2d");
    const segmentWidth = segment.end - segment.start;

    // Extract segment
    const segmentCanvas = document.createElement("canvas");
    segmentCanvas.width = segmentWidth;
    segmentCanvas.height = sourceCanvas.height;
    const segmentCtx = segmentCanvas.getContext("2d");
    segmentCtx.drawImage(
      sourceCanvas,
      segment.start,
      0,
      segmentWidth,
      sourceCanvas.height,
      0,
      0,
      segmentWidth,
      sourceCanvas.height
    );

    // Find bounding box of actual digit
    const segmentData = segmentCtx.getImageData(
      0,
      0,
      segmentWidth,
      sourceCanvas.height
    );
    const pixels = segmentData.data;

    let minX = segmentWidth,
      maxX = 0,
      minY = sourceCanvas.height,
      maxY = 0;
    for (let y = 0; y < sourceCanvas.height; y++) {
      for (let x = 0; x < segmentWidth; x++) {
        const idx = (y * segmentWidth + x) * 4;
        if (pixels[idx] > 50) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }

    if (maxX <= minX || maxY <= minY) return null;

    const contentWidth = maxX - minX + 1;
    const contentHeight = maxY - minY + 1;

    // MNIST preprocessing: fit to 20x20 box, then center in 28x28
    const maxDim = Math.max(contentWidth, contentHeight);
    const scale = 20.0 / maxDim;
    const scaledWidth = Math.round(contentWidth * scale);
    const scaledHeight = Math.round(contentHeight * scale);

    // Create 28x28 canvas (MNIST size)
    const mnistCanvas = document.createElement("canvas");
    mnistCanvas.width = 28;
    mnistCanvas.height = 28;
    const mnistCtx = mnistCanvas.getContext("2d");

    // Fill with black
    mnistCtx.fillStyle = "black";
    mnistCtx.fillRect(0, 0, 28, 28);

    // Center the scaled digit
    const offsetX = Math.round((28 - scaledWidth) / 2);
    const offsetY = Math.round((28 - scaledHeight) / 2);

    // Use better image smoothing
    mnistCtx.imageSmoothingEnabled = true;
    mnistCtx.imageSmoothingQuality = "high";

    // Draw centered and scaled digit
    mnistCtx.drawImage(
      segmentCanvas,
      minX,
      minY,
      contentWidth,
      contentHeight,
      offsetX,
      offsetY,
      scaledWidth,
      scaledHeight
    );

    return mnistCanvas;
  };

  const recognizeDigit = () => {
    if (!n) {
      alert("Network not initialized! Please train the network first.");
      return;
    }

    const canvas = canvasRef.current;
    const segments = segmentDigits(canvas);

    if (segments.length === 0) {
      alert("No digits detected! Please draw a number.");
      return;
    }

    const results = [];
    const segmentImages = [];
    const processedImages = [];

    for (const segment of segments) {
      const mnistCanvas = preprocessDigitForMNIST(canvas, segment);
      if (!mnistCanvas) continue;

      // Save for visualization
      const segmentCanvas = document.createElement("canvas");
      const segmentWidth = segment.end - segment.start;
      segmentCanvas.width = segmentWidth;
      segmentCanvas.height = canvas.height;
      const segmentCtx = segmentCanvas.getContext("2d");
      segmentCtx.drawImage(
        canvas,
        segment.start,
        0,
        segmentWidth,
        canvas.height,
        0,
        0,
        segmentWidth,
        canvas.height
      );
      segmentImages.push(segmentCanvas.toDataURL());
      processedImages.push(mnistCanvas.toDataURL());

      // Extract and normalize pixel data
      const mnistCtx = mnistCanvas.getContext("2d");
      const imageData = mnistCtx.getImageData(0, 0, 28, 28);
      const pixels = imageData.data;

      const inputs = [];
      for (let i = 0; i < pixels.length; i += 4) {
        const gray = pixels[i]; // R channel (grayscale)
        inputs.push((gray / 255.0) * 0.99 + 0.01);
      }

      // Query the network
      const outputs = n.query(inputs);
      const maxOutput = Math.max(...outputs);
      const label = outputs.indexOf(maxOutput);

      // Get top 3 predictions for debugging
      const predictions = outputs
        .map((val, idx) => ({ digit: idx, confidence: val }))
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3);

      results.push({
        digit: label,
        confidence: (maxOutput * 100).toFixed(1),
        allPredictions: predictions,
      });
    }

    if (results.length === 0) {
      alert("Could not process any digits. Please draw clearer.");
      return;
    }

    const fullNumber = results.map((r) => r.digit).join("");
    const avgConfidence = (
      results.reduce((sum, r) => sum + parseFloat(r.confidence), 0) /
      results.length
    ).toFixed(1);

    setPrediction(fullNumber);
    setConfidence(avgConfidence);
    setDigitSegments(
      results.map((r, i) => ({
        ...r,
        originalImage: segmentImages[i],
        processedImage: processedImages[i],
      }))
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-gray-800">
              Multi-Digit Handwritten Number Recognition with Improved Accuracy
            </h1>
            <p className="text-gray-600">
              Based on "Make Your Own Neural Network" by Tariq Rashid
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 mb-8">
            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Training</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Training Data (CSV)
                  </label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => loadMNISTData(e.target.files[0], true)}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Upload mnist_train.csv (60,000 examples)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Test Data (CSV)
                  </label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => loadMNISTData(e.target.files[0], false)}
                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Upload mnist_test.csv (10,000 examples)
                  </p>
                </div>

                <button
                  onClick={trainNetwork}
                  disabled={isTraining || !trainingData}
                  className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                  {isTraining ? "Training..." : "Train Network (5 epochs)"}
                </button>

                {isTraining && (
                  <div className="space-y-2">
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div
                        className="bg-indigo-600 h-4 rounded-full transition-all duration-300"
                        style={{ width: `${trainingProgress}%` }}
                      />
                    </div>
                    <p className="text-sm text-center text-gray-600">
                      {trainingProgress}% complete
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6">
              <h2 className="text-xl font-semibold mb-4">Testing</h2>

              <div className="space-y-4">
                <p className="text-sm text-gray-600">Network Configuration:</p>
                <ul className="text-sm space-y-1 text-gray-700">
                  <li>• Input: 784 (28×28 pixels)</li>
                  <li>• Hidden: 200 nodes</li>
                  <li>• Output: 10 (digits 0-9)</li>
                  <li>• Learning rate: 0.1</li>
                  <li>• Activation: Sigmoid</li>
                  <li>• Epochs: 5</li>
                </ul>

                <button
                  onClick={testNetwork}
                  disabled={!testData}
                  className="w-full bg-green-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                >
                  Test Network
                </button>

                {testAccuracy !== null && (
                  <div className="bg-white rounded-lg p-4 text-center">
                    <p className="text-sm text-gray-600 mb-1">Test Accuracy</p>
                    <p className="text-4xl font-bold text-green-600">
                      {testAccuracy}%
                    </p>
                    <p className="text-xs text-gray-500 mt-1">(Book: ~95%)</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6">
            <h2 className="text-xl font-semibold mb-4">Draw & Recognize</h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-gray-600 mb-3">
                  Draw digits with clear spacing (write large and bold):
                </p>
                <canvas
                  ref={canvasRef}
                  width={560}
                  height={280}
                  className="border-4 border-gray-300 rounded-lg cursor-crosshair bg-black w-full"
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                  onTouchStart={startDrawing}
                  onTouchMove={draw}
                  onTouchEnd={stopDrawing}
                />
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={clearCanvas}
                    className="flex-1 bg-gray-500 text-white py-2 px-4 rounded-lg font-semibold hover:bg-gray-600 transition-colors"
                  >
                    Clear
                  </button>
                  <button
                    onClick={recognizeDigit}
                    disabled={!n}
                    className="flex-1 bg-purple-600 text-white py-2 px-4 rounded-lg font-semibold hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                  >
                    Recognize
                  </button>
                </div>
              </div>

              <div className="flex flex-col items-center justify-center">
                {prediction !== null ? (
                  <div className="text-center w-full">
                    <p className="text-gray-600 mb-2">Result:</p>
                    <p className="text-7xl font-bold text-purple-600 mb-4 break-all">
                      {prediction}
                    </p>
                    <p className="text-lg text-gray-600 mb-4">
                      Confidence:{" "}
                      <span className="font-semibold">{confidence}%</span>
                    </p>

                    {digitSegments.length > 0 && (
                      <div className="bg-white rounded-lg p-4">
                        <p className="text-sm font-semibold text-gray-700 mb-3 text-center">
                          What the Network Sees:
                        </p>
                        <div className="space-y-3">
                          {digitSegments.map((seg, idx) => (
                            <div
                              key={idx}
                              className="bg-gray-50 rounded-lg p-3 border border-gray-200"
                            >
                              <div className="flex items-center gap-3">
                                <div>
                                  <p className="text-xs text-gray-500 mb-1">
                                    Your Drawing
                                  </p>
                                  <img
                                    src={seg.originalImage}
                                    alt={`Original ${idx}`}
                                    className="w-16 h-16 border border-gray-300"
                                  />
                                </div>
                                <div className="text-xl text-gray-400">→</div>
                                <div>
                                  <p className="text-xs text-gray-500 mb-1">
                                    MNIST Format
                                  </p>
                                  <img
                                    src={seg.processedImage}
                                    alt={`Processed ${idx}`}
                                    className="w-16 h-16 border border-gray-300"
                                  />
                                </div>
                                <div className="text-xl text-gray-400">=</div>
                                <div className="text-center">
                                  <p className="text-3xl font-bold text-purple-600">
                                    {seg.digit}
                                  </p>
                                  <p className="text-xs text-gray-500">
                                    {seg.confidence}%
                                  </p>
                                </div>
                              </div>
                              <div className="mt-2 text-xs text-gray-600">
                                Top 3:{" "}
                                {seg.allPredictions
                                  .map(
                                    (p) =>
                                      `${p.digit}(${(
                                        p.confidence * 100
                                      ).toFixed(0)}%)`
                                  )
                                  .join(", ")}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center text-gray-400">
                    <p className="mb-2">Draw a number and click "Recognize"</p>
                    <p className="text-sm">
                      Tips: Write large, bold, with spacing
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="mt-6 bg-amber-50 border border-amber-200 rounded-lg p-4">
            <h3 className="font-semibold text-amber-900 mb-2">
              Tips for Better Recognition:
            </h3>
            <ul className="text-sm text-amber-800 space-y-1">
              <li>
                • <strong>Write LARGE</strong> - fill the vertical space
              </li>
              <li>
                • <strong>Write BOLD</strong> - use thick strokes
              </li>
              <li>
                • <strong>Leave GAPS</strong> - space between digits
              </li>
              <li>
                • <strong>Center vertically</strong> - middle of the canvas
              </li>
              <li>• Train the network first before testing recognition</li>
              <li>• Check "What the Network Sees" to debug issues</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MNISTDigitRecognition;
