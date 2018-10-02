<template>
  <div id="app">
    <div class="title-banner">
      <img src="/static/Hello Picture.jpg" alt="">
    </div>
    <div class="uk-container uk-container-large uk-text-center uk-margin">

      <div class="uk-child-width-1-2@m uk-margin-remove" uk-grid>

        <!-- 1. Paste URL -->
        <h3>Paste the URL to image</h3>

        <!-- 2. Upload image -->
        <div>
          <h3>Paste the URL to image</h3>
          <div class="uk-animation-toggle" uk-form-custom>
            <input type="file" @change="onPickFile">
            <button class="uk-button uk-button-secondary uk-animation-shake" type="button" tabindex="-1">Upload Image</button>
          </div>
        </div>
      </div>
      <div class="uk-child-width-1-2@m uk-margin-remove" uk-grid>
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

.uk-placeholder {
  border: 3px dashed #000000;
}

.title-banner {
  margin: 20px 20px;
}
</style>
