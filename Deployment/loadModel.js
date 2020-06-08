class loadModel{
    load(){
        const MODEL_URL = 'http://127.0.0.1:8887/jsModel/model.json';
        const model = tf.loadLayersModel(MODEL_URL);
        return model
    }

    async print(){
        const model = await this.load(); 
        console.log(model.summary())
        alert('Model is Loaded')
    }



}



