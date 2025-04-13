<template>
  <div id="app">
    <h1>手写数字识别</h1>
    <div class="canvas-container">
      <canvas
        ref="canvas"
        width="280"
        height="280"
        @mousedown="startDrawing"
        @mousemove="draw"
        @mouseup="stopDrawing"
        @mouseleave="stopDrawing"
      ></canvas>
    </div>
    <button @click="clearCanvas">清空</button>
    <button @click="submitDrawing">识别</button>
    <p v-if="prediction !== null">预测结果：{{ prediction }}</p>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      drawing: false,
      context: null,
      prediction: null,
    };
  },
  mounted() {
    const canvas = this.$refs.canvas;
    this.context = canvas.getContext('2d');
    this.context.fillStyle = 'black';
    this.context.fillRect(0, 0, canvas.width, canvas.height); // 初始化为黑色背景
    this.context.strokeStyle = 'white';
    this.context.lineWidth = 10;
  },
  methods: {
    startDrawing(event) {
      this.drawing = true;
      this.context.beginPath();
      this.context.moveTo(event.offsetX, event.offsetY);
    },
    draw(event) {
      if (!this.drawing) return;
      this.context.lineTo(event.offsetX, event.offsetY);
      this.context.stroke();
    },
    stopDrawing() {
      this.drawing = false;
      this.context.closePath();
    },
    clearCanvas() {
      this.context.fillStyle = 'black';
      this.context.fillRect(0, 0, 280, 280);
    },
    async submitDrawing() {
      // 获取图像数据
      const imageData = this.context.getImageData(0, 0, 280, 280);
      const grayscaleImage = [];
      for (let i = 0; i < imageData.data.length; i += 4) {
        // 将图像转换为灰度值（0-1之间）
        const grayscale = imageData.data[i] / 255;
        grayscaleImage.push(grayscale);
      }

      // 将图像调整为 28x28
      const resizedImage = [];
      for (let y = 0; y < 28; y++) {
        const row = [];
        for (let x = 0; x < 28; x++) {
          const pixelIndex = (y * 10) * 280 + (x * 10);
          row.push(grayscaleImage[pixelIndex]);
        }
        resizedImage.push(row);
      }

      try {
        // 调用后端 API
        const response = await axios.post('http://127.0.0.1:5000/predict', {
          image: resizedImage,
        });
        this.prediction = response.data.prediction;
      } catch (error) {
        console.error('Error during prediction:', error);
      }
    },
  },
};
</script>

<style>
#app {
  text-align: center;
  margin-top: 20px;
}

.canvas-container {
  display: inline-block;
  border: 1px solid #000;
}

canvas {
  background-color: black;
  cursor: crosshair;
}

button {
  margin: 10px;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
}
</style>
