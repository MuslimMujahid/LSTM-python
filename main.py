import pandas as pd
from LEMBUT import layers
from LEMBUT.model import Sequential


def main():
    # Load dataset
    dataset_train = pd.read_csv('./dataset/bitcoin_price_Training - Training.csv')
    # dataset = pd.read_csv('./dataset/bitcoin_price_1week_Test - Test.csv')

    # Preprocessing
    dataset_train = dataset_train.iloc[0:32, :]
    dataset_train = dataset_train.iloc[::-1].reset_index(drop=True)
    dataset_train = dataset_train.drop(columns=['Date'])
    dataset_train['Volume'] = dataset_train['Volume'].apply(
        lambda x: x.replace(',', ''))
    dataset_train['Market Cap'] = dataset_train['Market Cap'].apply(
        lambda x: x.replace(',', ''))
    dataset_train['Volume'] = pd.to_numeric(dataset_train['Volume'])
    dataset_train['Market Cap'] = pd.to_numeric(dataset_train['Market Cap'])
    # print(dataset_train)

    # Create model
    model = Sequential()
    model.add(layers.LSTM(64, input_shape=(1, 4, 6), name='lstm_1'))
    prediction = model.predict(dataset_train)
    print(prediction)
    model.summary()


if __name__ == '__main__':
    main()
