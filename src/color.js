import * as tf from '@tensorflow/tfjs'
var space = require('color-space')

export function matrixMultiply (a, b) {
  /**
    * https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product
    */
  var H = a.shape[0]
  var W = a.shape[1]
  var c = tf.tensor(new Float32Array(H * W * 2), [H, W, 2])
  var buffer = c.buffer()
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      for (let g = 0; g < 2; g++) {
        for (let k = 0; k < 313; k++) {
          var value = c.get(i, j, g) + a.get(i, j, k) * b.get(k, g)
          buffer.set(value, i, j, g)
        }
      }
    }
  }
  return c
}

export var concatenate = function (l, ab) {
  var H = ab.shape[0]
  var W = ab.shape[1]
  var output = tf.tensor(new Float32Array(H * W * 3), [H, W, 3])
  var buffer = output.buffer()
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      buffer.set(l.get(i, j, 0), i, j, 0)
      buffer.set(ab.get(i, j, 0), i, j, 1)
      buffer.set(ab.get(i, j, 1), i, j, 2)
    }
  }
  return output
}

export var rgb2lab = function (rgb) {
  var H = rgb.shape[0]
  var W = rgb.shape[1]
  var lab = tf.tensor(new Float32Array(H * W * 3), [H, W, 3])
  var buffer = lab.buffer()
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      var r = rgb.get(i, j, 0) * 255
      var g = rgb.get(i, j, 1) * 255
      var b = rgb.get(i, j, 2) * 255
      var values = space.rgb.lab([r, g, b])
      buffer.set(values[0], i, j, 0)
      buffer.set(values[1], i, j, 1)
      buffer.set(values[2], i, j, 2)
    }
  }
  return lab
}

export var lab2rgb = function (lab) {
  var H = lab.shape[0]
  var W = lab.shape[1]
  var rgb = tf.tensor(new Float32Array(H * W * 3), [H, W, 3])
  const buffer = rgb.buffer()
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      var l = lab.get(i, j, 0)
      var a = lab.get(i, j, 1)
      var b = lab.get(i, j, 2)
      var values = space.lab.rgb([l, a, b])
      buffer.set(values[0], i, j, 0)
      buffer.set(values[1], i, j, 1)
      buffer.set(values[2], i, j, 2)
    }
  }
  return rgb
}
