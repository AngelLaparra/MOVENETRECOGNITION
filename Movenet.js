const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4';
const EXAMPLE_IMG = document.getElementById('SnapCanvas');
const CROPPED_CANVAS = document.getElementById('croppedCanvas');
let movenet = undefined;
async function loadAndRunModel() {
movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);
let cropStartPoint = [15, 170, 0];
let cropSize = [345, 345, 3];
let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);
await tf.browser.toPixels(croppedTensor, CROPPED_CANVAS);
let resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192],
true).toInt();
let inputTensor = tf.expandDims(resizedTensor);
let tensorOutput = movenet.predict(inputTensor);
let arrayOutput = await tensorOutput.array();
console.log("Salida de MoveNet:", arrayOutput);
if (arrayOutput.length > 0 && arrayOutput[0][0]) {
let keypoints = arrayOutput[0][0];

drawSkeleton(keypoints, CROPPED_CANVAS, cropSize[0] / 192);
} else {
console.error("Error: No se encontraron keypoints en la salida del modelo.");
}
imageTensor.dispose();
croppedTensor.dispose();
resizedTensor.dispose();
inputTensor.dispose();
tensorOutput.dispose();
}
function drawSkeleton(keypoints, canvas, scale) {
    const ctx = canvas.getContext("2d");
    ctx.lineWidth = 2;
    const connections = [
        [0, 1], [0, 2], // Nariz a ojos
        [1, 3], [2, 4], // Ojos a orejas
        [5, 7], [6, 8], // Hombros a codos
        [7, 9], [8, 10], // Codos a muñecas
        [11, 13], [12, 14], // Caderas a rodillas
        [13, 15], [14, 16], // Rodillas a tobillos
        [5, 11], [6, 12], // Hombros a caderas
    ];
    connections.forEach(([i, j]) => {
        let x1 = keypoints[i][1] * 192 * scale;
        let y1 = keypoints[i][0] * 192 * scale;
        let x2 = keypoints[j][1] * 192 * scale;
        let y2 = keypoints[j][0] * 192 * scale;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    });

    keypoints.forEach((point, index) => {
        let x = point[1] * 192 * scale; // Se intercambia x <-> y y se reescala
        let y = point[0] * 192 * scale;

        // Verificar si las manos están por encima de la cabeza
        let isHandAboveHead = (index === 9 || index === 10) && (y < keypoints[0][0] * 192 * scale);

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI); // Dibujar un círculo en cada punto clave
        ctx.fillStyle = isHandAboveHead ? "red" : "green"; // Pintar en rojo si las manos están por encima de la cabeza, en verde en caso contrario
        ctx.fill();
        console.log(x);
        console.log(y);
    });
}