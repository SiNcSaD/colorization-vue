<template>
  <div id="app">
    <div class="title-banner uk-text-center">
      <img src="/static/Hello Picture.jpg" alt="">
    </div>
    <div class="uk-container uk-container-large uk-text-center">
      <div>
        <h1 class="rainbow uk-margin-remove">Colorization</h1>
        <h3 style="margin: 20px 0px">Convert Grayscale Images to Color Images Based on Deep Learning</h3>
      </div>
      <div class="uk-child-width-1-2@m uk-child-width-1-1@s uk-flex-middle uk-margin-remove" uk-grid>
        <!-- 1. Paste URL -->
        <div class="uk-padding-remove">
          <h3 class="subtitle">Paste the URL to an image</h3>
          <div uk-margin>
            <div class="uk-inline uk-width-1-2">
              <span class="uk-form-icon" uk-icon="icon: link"></span>
              <input class="uk-input" type="text" placeholder="http://" v-model="onlineURL">
            </div>
            <button class="uk-button uk-button-primary" v-on:click="onClickSubmit">Submit</button>
          </div>
        </div>

        <!-- 2. Upload image -->
        <div class="uk-padding-remove">
          <h3 class="subtitle">Click the button to upload an image</h3>
          <div class="uk-animation-toggle" uk-form-custom>
            <input type="file" @change="onPickFile">
            <button class="uk-button uk-button-secondary uk-animation-shake" type="button" tabindex="-1">
              <span><i class="fa fa-upload uk-margin-small-right" aria-hidden="true"></i>Upload</span>
            </button>
          </div>
        </div>
      </div>

      <div v-if="isWait === true">
        <img src="/static/Spin-1s-200px.gif">
      </div>

      <!-- Show input and output -->
      <div v-if="isOK === true" id="result-compare" class="uk-flex uk-flex-center uk-margin-medium-top">
        <image-compare :before="imageDst" :after="imageSrc">
          <i class="fa fa-angle-left slider-compare" aria-hidden="true" slot="icon-left"></i>
          <i class="fa fa-angle-right slider-compare" aria-hidden="true" slot="icon-right"></i>
        </image-compare>
      </div>

      <div class="uk-child-width-1-2@m uk-margin-top" style="display:none" uk-grid>
        <div class="uk-padding-remove">
          <img id="img-src" :src="imageSrc" max-height="300px">
        </div>
        <div class="uk-padding-remove">
          <canvas id="canvas-dst" width="224" height="224"></canvas>
        </div>
      </div>

    </div>
    <div class="uk-section uk-section-secondary uk-margin-medium-top bottom_info">
      <div class="uk-container uk-text-center">
        <h3>國立台北科技大學 - 電機工程系 - 高效能計算與深度學習研究室</h3>
        <h3 class="uk-margin-remove">National Taipei University of Technology - Department of Electrical Engineering</h3>
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
  this.msgState = 'Loading model...'
  var model = await tf.loadModel(MODEL_PATH)
  console.log('Warmup the model')
  this.msgState = 'Warmup the model'
  model.predict(tf.zeros([1, 224, 224, 1])).dispose()
  return model
}

export default {
  name: 'App',
  data () {
    return {
      imageSrc: '/static/after.jpg',
      imageDst: '/static/before.jpg',
      isOK: false,
      isWait: false,
      msgState: '.............',
      onlineURL: ''
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
    onClickSubmit: function (event) {
      this.imageSrc = '/static/ILSVRC2012_val_00046524.JPEG'
      this.inference()
      this.clearMessage()
    },
    clearMessage () {
      this.onlineURL = ''
    },
    inference () {
      this.isWait = true
      loadModel().then(model => {
        console.log('Predicting...')
        this.msgState = 'Predicting...'

        /**
         * Get element of html <img>
         */
        var imgSrcElement = document.getElementById('img-src')
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
        var c = document.getElementById('canvas-dst')
        var ctx = c.getContext('2d')
        c.height = H
        c.width = W
        const imgResultT = tf.transpose(imgResult, [1, 0, 2])
        var canvasResult = savePixels(ndarray(imgResultT.dataSync(), imgResultT.shape), 'canvas')

        /**
         * Show result to HTML
         */
        ctx.drawImage(canvasResult, 0, 0, W, H)
        this.imageDst = c.toDataURL()
        this.isOK = true
        this.isWait = false
        // document.getElementById('result-compare').style.display = 'block'

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
.image-compare {
  width: 500px;
}
.uk-placeholder {
  border: 3px dashed #000000;
}
.title-banner {
  margin: 20px 20px;
}
.slider-compare {
  font-size: 32px;
  color: slateblue;
}
.image-compare-handle[data-v-2aa9daa6] {
  color: slateblue;
}
.bottom_info h3 {
  font-family: SimSun, Microsoft JhengHei;
}
.rainbow {
  background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet);
  color: transparent;
  -webkit-background-clip: text;
  background-clip: text;
  font-size: 50px;
  display:inline-block;
}
.subtitle {
  font-size: 32px;
}
</style>
