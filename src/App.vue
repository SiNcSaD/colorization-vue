<template>
  <div id="app">
    <div class="title-banner">
      <img src="/static/Hello Picture.jpg" alt="">
    </div>
    <div class="uk-container uk-container-large uk-text-center uk-margin">
      <div>
        <h1>Colorization</h1>
        <h4 class="uk-margin">Convert Grayscale Images to Color Images Based on Deep Learning</h4>
      </div>
      <div class="uk-child-width-1-2@m uk-margin-remove uk-flex-middle" uk-grid>
        <!-- 1. Paste URL -->
        <div class="uk-padding-remove">
          <h3>Paste the URL to an image</h3>
          <form>
            <div uk-margin>
              <div class="uk-inline uk-width-1-2">
                <span class="uk-form-icon" uk-icon="icon: link"></span>
                <input class="uk-input" type="text" placeholder="http://">
              </div>
              <button class="uk-button uk-button-default">Submit</button>
            </div>
          </form>
        </div>

        <!-- 2. Upload image -->
        <div class="uk-padding-remove">
          <h3>Click the button to upload an image</h3>
          <div class="uk-animation-toggle" uk-form-custom>
            <input type="file" @change="onPickFile">
            <button class="uk-button uk-animation-shake upload-button" type="button" tabindex="-1">Upload Image</button>
          </div>
        </div>
      </div>

      <!-- Show input and output -->
      <div class="uk-child-width-1-2@m uk-margin-top" uk-grid>
        <div class="uk-padding-remove">
          <img id="img_Src" :src="imageSrc" max-height="300px">
        </div>

        <div class="uk-padding-remove">
          <canvas id="canvas_Out" width="224" height="224"></canvas>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import { ptsInHull } from './pts'
import { matrixMultiply, concatenate, rgb2lab, lab2rgb } from './color'

const MODEL_PATH = '/static/web_tfjs_model/model.json'

var ndarray = require('ndarray')
var savePixels = require('save-pixels')

const loadModel = async () => {
  console.log('Loading model...')
  var model = await tf.loadModel(MODEL_PATH)
  console.log('Warmup the model')
  model.predict(tf.zeros([1, 224, 224, 1])).dispose()
  return model
}

export default {
  name: 'App',
  data () {
    return {
      imageSrc: null
    }
  },
  created: function () {
    // initComparisons()
  },
  methods: {
    onPickFile (e) {
      var files = e.target.files || e.dataTransfer.files
      if (!files.length) return
      this.onFilePicked(files[0])
    },
    onFilePicked (file) {
      var reader = new FileReader()
      reader.onload = e => {
        this.imageSrc = e.target.result
      }
      reader.readAsDataURL(file)

      this.inference()
    },
    inference () {
      loadModel().then(model => {
        console.log('Predicting...')
        /**
         * Get element of html <img>
         */
        var imgSrcElement = document.getElementById('img_Src')
        var H = imgSrcElement.height
        var W = imgSrcElement.width
        this.imgSrcHeight = H
        this.imgSrcWidth = W

        /**
         * Loading image and normalization
         */
        var img = tf.fromPixels(imgSrcElement).toFloat()
        var imgRGB = img.div(255.0)
        const predication = tf.tidy(() => {
          /**
          * Image pre-processing and prediction
          */
          var imgRgbResize = tf.image.resizeBilinear(imgRGB, [224, 224])
          var imgLabResize = rgb2lab(imgRgbResize)
          var imgL = imgLabResize.slice([0, 0, 0], [224, 224, 1])
          imgL = imgL.sub(50)
          return model.predict(imgL.reshape([1, 224, 224, 1]))
        })

        /**
         * I don't know
         */
        const predicationRH = predication.mul(2.606)
        const classRH = predicationRH.softmax().reshape([56, 56, 313])
        const cc = ptsInHull()

        /**
         * matrix multiply
         */
        const dataAB = matrixMultiply(classRH, cc)

        /**
         * Resize (Height/4, Width/4, 2) to (Height, Width, 2)
         */
        const resizeAB = tf.image.resizeBilinear(dataAB, [H, W])

        /**
         * Concatenate dataL and dataAB
         */
        const dataLab = rgb2lab(imgRGB)
        var imgLAB = concatenate(dataLab, resizeAB)

        /**
         * Convert lab to rgb
         */
        var imgResult = lab2rgb(imgLAB)
        /**
         * Draw result image
         */
        var c = document.getElementById('canvas_Out')
        var ctx = c.getContext('2d')
        c.height = H
        c.width = W
        const imgResultT = tf.transpose(imgResult, [1, 0, 2])
        var canvasResult = savePixels(ndarray(imgResultT.dataSync(), imgResultT.shape), 'canvas')
        ctx.drawImage(canvasResult, 0, 0, W, H)

        /**
         * dispose tensor variable
         */
        img.dispose()
        imgRGB.dispose()
        predicationRH.dispose()
        classRH.dispose()
        resizeAB.dispose()
        dataLab.dispose()
        imgLAB.dispose()
        imgResult.dispose()
        imgResultT.dispose()
      })
    }
  }
}
</script>

<style>
html {
  background: #F6F9FA;
}
h1 {
  font-weight: bold;
}
.uk-placeholder {
  border: 3px dashed #000000;
}
.title-banner {
  margin: 20px 20px;
}
.upload-button {
  background: #657c89;
  color: #ffffff;
}
* {box-sizing: border-box;}
.img-comp-container {
  position: relative;
  height: 200px; /*should be the same height as the images*/
}
.img-comp-img {
  position: absolute;
  width: auto;
  height: auto;
  overflow: hidden;
}
.img-comp-img img {
  display: block;
  vertical-align: middle;
}
.img-comp-slider {
  position: absolute;
  z-index: 9;
  cursor: ew-resize;
  /*set the appearance of the slider:*/
  width: 40px;
  height: 40px;
  background-color: #2196F3;
  opacity: 0.7;
  border-radius: 50%;
}
</style>
