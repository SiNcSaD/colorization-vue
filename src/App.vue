<template>
  <div id="app">
    <div class="title-banner">
      <img src="/static/Hello Picture.jpg" alt="">
    </div>
    <div class="uk-container uk-container-middle uk-text-center uk-margin">

      <div class="uk-child-width-1-2@m uk-flex-midel uk-margin-remove" uk-grid>

        <!-- 1. Paste URL -->
        <h3>網址區</h3>

        <!-- 2. Upload image -->
        <div>
          <div class="uk-placeholder uk-text-center">
            <h3>Select a grayscale image or color image.</h3>
          </div>
          <div uk-form-custom>
            <input type="file" @change="onFileChange">
            <button class="uk-button uk-button-secondary" type="button" tabindex="-1">Upload</button>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
html {
  background: #F6F9FA;
}
.uk-placeholder {
  border: 3px dashed #9f00bb;
}
.title-banner {
  margin: 20px 20px;
}
</style>
