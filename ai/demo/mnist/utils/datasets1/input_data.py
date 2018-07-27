import utils.datasets1.utils as utils

class Data_Set:
    def __init__(self, data):
        self.images = data[0]
        self.labels = data[1]
        self.cur = 0
        self.num_examples = self.images.shape[0]
        #self.num_features = self.images.shape[1]
    def next_batch(self, batch_size):
        start = self.cur
        end = start + batch_size
        self.cur = end
        if end >= self.num_examples:
            end = self.num_examples
            self.cur = 0
        #print("data: [%d, %d), %d" % (start, end, end - start))
        return self.images[start:end, :], self.labels[start:end, :]
            
class MNIST_Data:
    def __init__(self, train, val, test):
        self.train = Data_Set(train)
        self.validation = Data_Set(val)
        self.test = Data_Set(test)

def read_data_sets(path, one_hot):
    utils.download_mnist(path)
    train, val, test = utils.read_mnist(path, one_hot)
    mnist = MNIST_Data(train, val, test)
    return mnist


def test():
    batch_size = 128
    mnist = read_data_sets('data/mnist', one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    n_batches = int(mnist.test.num_examples/batch_size) + 1
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        print(X_batch.shape, Y_batch.shape)

if __name__ == '__main__':
    test()
    
