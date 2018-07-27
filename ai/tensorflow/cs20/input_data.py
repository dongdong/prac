import utils

class Data_Set:
    def __init__(self, data):
        self.X = data[0]
        self.Y = data[1]
        self.cur = 0
        self.num_examples = self.X.shape[0]
        #self.num_features = self.X.shape[1]
    def next_batch(self, batch_size):
        start = self.cur
        end = start + batch_size
        self.cur = end
        if end > self.num_examples:
            end = self.num_examples
            self.cur = 0
        #print("data: [%d, %d), %d" % (start, end, end - start))
        return self.X[start:end, :], self.Y[start:end, :]
            
class MNIST_Data:
    def __init__(self, train, test):
        self.train = Data_Set(train)
        self.test = Data_Set(test)

def read_data_sets(path, one_hot):
    utils.download_mnist(path)
    train, val, test = utils.read_mnist(path, one_hot)
    mnist = MNIST_Data(train, test)
    return mnist


def test():
    batch_size = 128
    mnist = read_data_sets('data/mnist', one_hot=True)
    print(mnist.train.X.shape, mnist.train.Y.shape)
    print(mnist.test.X.shape, mnist.test.Y.shape)
    n_batches = int(mnist.test.num_examples/batch_size) + 1
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        print(X_batch.shape, Y_batch.shape)

if __name__ == '__main__':
    test()
    
