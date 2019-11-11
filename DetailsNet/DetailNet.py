import datetime
from network import inference
from GuidedFilter import guided_filter


class DetailNet:
    
    def __init__(self, img_shape=(512, 512, 3)):
        self.img_shape = img_shape
        
        self.network = inference(img_shape, True)
        self.network.compile('rmsprop', loss=['mse'])
        
    
    def train(self, data_loader, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()
        
        for epoch in range(epochs):
            for batch_i, (img_input, target_input) in enumerate(data_loader.load_batch(batch_size)):

                base = guided_filter(img_input, img_input, 15, 1, nhwc=True)
                detail_input = img_input - base
                
                loss = self.network.train_on_batch(
                    [img_input, detail_input], 
                    target_input
                )

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [loss: %f] time: %s " % 
                      (epoch, epochs,
                       batch_i, data_loader.n_batches,
                       loss,
                       elapsed_time))
