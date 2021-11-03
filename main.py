import pandas as pd
from LEMBUT import layers
from LEMBUT.model import Sequential


def main():
    # Load dataset
    # dataset = pd.read_csv('./dataset/bitcoin_price_Training - Training.csv')
    dataset = pd.read_csv('./dataset/bitcoin_price_1week_Test - Test.csv')

    # Preprocessing
    dataset = dataset.drop(columns=['Date'])
    dataset['Volume'] = dataset['Volume'].apply(
        lambda x: x.replace(',', ''))
    dataset['Market Cap'] = dataset['Market Cap'].apply(
        lambda x: x.replace(',', ''))
    dataset['Volume'] = pd.to_numeric(dataset['Volume'])
    dataset['Market Cap'] = pd.to_numeric(dataset['Market Cap'])

    # Create model
    model = Sequential()
    model.add(layers.LSTM(input_shape=(1, 4, 6)))
    model.predict(dataset)


if __name__ == '__main__':
    main()
