const fileInput = document.querySelector("#file-input");
const chooseButton = document.querySelector("#choose-button");
const analyzeButton = document.querySelector("#analyze-button");
const dropArea = document.querySelector("#drop-area");
const preview = document.querySelector("#preview");
const photoEmpty = document.querySelector("#photo-empty");
const statusText = document.querySelector("#status-text");
const modelStatus = document.querySelector("#model-status");
const resultClass = document.querySelector("#result-class");
const confidence = document.querySelector("#confidence");
const resultMessage = document.querySelector("#result-message");
const healthyValue = document.querySelector("#healthy-value");
const unhealthyValue = document.querySelector("#unhealthy-value");
const healthyFill = document.querySelector("#healthy-fill");
const unhealthyFill = document.querySelector("#unhealthy-fill");

let selectedFile = null;
let previewUrl = null;
let modelPromise = null;
let analyzing = false;

function setModelStatus(state, message) {
  modelStatus.dataset.state = state;
  statusText.textContent = message;
}

function setMessage(message, isError = false) {
  resultMessage.textContent = message;
  resultMessage.style.color = isError ? "#a13d34" : "";
}

function makeLayer(description, index, inputShape) {
  const config = description.config;
  const firstInput = index === 0 ? { inputShape } : {};

  switch (description.class_name) {
    case "Conv2D":
      return tf.layers.conv2d({
        ...firstInput,
        filters: config.filters,
        kernelSize: config.kernel_size,
        strides: config.strides,
        padding: config.padding,
        activation: config.activation,
        useBias: config.use_bias,
      });
    case "BatchNormalization":
      return tf.layers.batchNormalization({
        axis: -1,
        epsilon: config.epsilon,
        center: config.center,
        scale: config.scale,
      });
    case "MaxPooling2D":
      return tf.layers.maxPooling2d({
        poolSize: config.pool_size,
        strides: config.strides || config.pool_size,
        padding: config.padding,
      });
    case "Flatten":
      return tf.layers.flatten();
    case "Dense":
      return tf.layers.dense({
        units: config.units,
        activation: config.activation,
        useBias: config.use_bias,
      });
    default:
      throw new Error(`Capa no soportada: ${description.class_name}`);
  }
}

async function fetchWeights(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`No se pudieron cargar los pesos (${response.status}).`);

  const totalBytes = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !totalBytes) return response.arrayBuffer();

  const reader = response.body.getReader();
  const chunks = [];
  let receivedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedBytes += value.length;
    const downloaded = (receivedBytes / 1024 / 1024).toFixed(1);
    const total = (totalBytes / 1024 / 1024).toFixed(1);
    setModelStatus("loading", `Cargando modelo ${downloaded} / ${total} MB`);
  }

  const bytes = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes.buffer;
}

async function loadModel() {
  const manifestResponse = await fetch("./vercel_model/model.json");
  if (!manifestResponse.ok) throw new Error("No se encontro el manifiesto del modelo.");
  const manifest = await manifestResponse.json();
  const weightsBuffer = await fetchWeights("./vercel_model/weights.bin");
  const weights = new Float32Array(weightsBuffer);
  const model = tf.sequential();

  manifest.layers.forEach((description, index) => {
    model.add(makeLayer(description, index, manifest.input_shape));
  });

  const weightedLayers = model.layers.filter((layer) => layer.weights.length > 0);
  if (weightedLayers.length !== manifest.weighted_layers.length) {
    throw new Error("La estructura de pesos no coincide con la arquitectura.");
  }

  manifest.weighted_layers.forEach((layerSpec, layerIndex) => {
    const tensors = layerSpec.weights.map((weight) => {
      const values = weights.subarray(weight.offset, weight.offset + weight.length);
      return tf.tensor(values, weight.shape, "float32");
    });
    weightedLayers[layerIndex].setWeights(tensors);
    tensors.forEach((tensor) => tensor.dispose());
  });

  return { model, manifest };
}

function getModel() {
  if (!modelPromise) {
    modelPromise = loadModel().then((loaded) => {
      setModelStatus("ready", "Modelo cargado");
      return loaded;
    }).catch((error) => {
      modelPromise = null;
      setModelStatus("error", "No se pudo cargar el modelo");
      throw error;
    });
  }
  return modelPromise;
}

function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setMessage("Selecciona un archivo de imagen valido.", true);
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  photoEmpty.hidden = true;
  analyzeButton.disabled = false;
  resultClass.textContent = "Lista para analizar";
  confidence.textContent = "";
  setMessage("La primera carga del modelo puede tardar unos segundos.");
}

async function analyzeImage() {
  if (!selectedFile || analyzing) return;

  analyzing = true;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analizando...";
  setMessage("Preparando la inferencia en este navegador.");

  try {
    const { model, manifest } = await getModel();
    const [height, width] = manifest.input_shape;
    const probabilityUnhealthy = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(preview).toFloat().expandDims(0);
      const resized = tf.image.resizeBilinear(pixels, [height, width]);
      const input = resized.mul(manifest.input_scale);
      return model.predict(input);
    });
    const probability = (await probabilityUnhealthy.data())[0];
    probabilityUnhealthy.dispose();

    const healthy = 1 - probability;
    const isUnhealthy = probability >= 0.5;
    const healthyPercent = healthy * 100;
    const unhealthyPercent = probability * 100;
    resultClass.textContent = isUnhealthy ? "Unhealthy" : "Healthy";
    confidence.textContent = `Confianza: ${(Math.max(healthy, probability) * 100).toFixed(1)}%`;
    healthyValue.textContent = `${healthyPercent.toFixed(1)}%`;
    unhealthyValue.textContent = `${unhealthyPercent.toFixed(1)}%`;
    healthyFill.style.width = `${healthyPercent}%`;
    unhealthyFill.style.width = `${unhealthyPercent}%`;
    setMessage("Resultado generado localmente con la CNN.");
  } catch (error) {
    console.error(error);
    setMessage(error.message || "Ocurrio un error al analizar la imagen.", true);
    resultClass.textContent = "Error de analisis";
  } finally {
    analyzing = false;
    analyzeButton.disabled = !selectedFile;
    analyzeButton.textContent = "Analizar hoja";
  }
}

chooseButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
analyzeButton.addEventListener("click", analyzeImage);
dropArea.addEventListener("click", () => fileInput.click());
dropArea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
for (const eventName of ["dragenter", "dragover"]) {
  dropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropArea.dataset.dragging = "true";
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    delete dropArea.dataset.dragging;
  });
}
dropArea.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
