import pickle

from aes.ga import LifeBook, execute


def main():
    NUM_GENERATIONS = 50

    exp = execute(NUM_GENERATIONS, 20)
    pickle.dump(exp, open('artifacts/exp5.pkl', 'wb'))

    obj: LifeBook = pickle.load(open('artifacts/exp20.pkl', 'rb'))

    print('wait')



if __name__ == "__main__":
    main()
