from Organization import Org
from NeuralNet import NeuralNet
def main():
    """ Kieran Ringel
    For each data set three lines are run in main.
    The first creates an instance of Org with the arguments being the data file name, an array of rows with header
    information to be removed, and the column location of the class so that all the classes can be put in the same column.
    The second line takes the instance of Org and calls the open method, returning the pandas dataframe of the file.
    The third line creates an instance of ProcessData, the arguments are the dataframe created in Org.open(), classification or
    regression, the type (none, edited, condensed, reducedmed, reducedmean), and an array of the columns with discrete values."""

    print('Breast Cancer')
    cancer = Org('Data/breast-cancer-wisconsin.data', [-1], -1, [-1])
    df = cancer.open()
     ##NN(file, number hidden layers, number hidden nodes per layer)
    NeuralNet(df, 1, 5, 'classification')
    print(df)

    # print('glass')
    # glass = Org('Data/glass.data', [-1], -1, [-1])
    # df = glass.open()
    # print(df)

    # print('soybean')
    # soybean = Org('Data/soybean-small.data', [-1], -1, [-1])
    # df = soybean.open()
    # print(df)

    # print('abalone')
    # abalone = Org('Data/abalone.data', [-1], -1, [0])
    # df = abalone.open()
    # print(df)

##    print('machine')
##    machine = Org('Data/machine.data', [-1], -1, [-1])
##    df = machine.open()
##    NeuralNet(df, 2, 3, 'regression')
##    print(df)

    # print('forest')
    # forest = Org('Data/forestfires.data', [0], -1, [2,3])
    # df = forest.open()
    # print(df)

if __name__ == '__main__':
    main()
