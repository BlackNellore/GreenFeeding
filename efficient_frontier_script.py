import pandas
import sys


class Data:
    _data: pandas.DataFrame
    results: pandas.DataFrame = None
    filename: str = None
    x_column: str = None
    y_column: str = None

    def __init__(self, filename, x, y, sheet=None):
        self.x_column = x
        self.y_column = y
        self.filename = filename
        if ".xlsx" in filename:
            excel_file = pandas.ExcelFile(filename)
            self._data = pandas.read_excel(excel_file, sheet)
        elif ".csv" in filename:
            self._data = pandas.read_csv(filename)

        self._data = self._data.sort_values(by=x).reset_index()

    def eliminate(self):
        total_points = self._data[self.x_column].size
        elimination_left = set()
        for i in range(1, total_points):
            if i in elimination_left:
                continue
            for k in range(i):
                if k in elimination_left:
                    continue
                if self._data[self.y_column][i] < self._data[self.y_column][k]:
                    elimination_left.add(i)
                    continue
        elimination_right = set()
        for i in range(1, total_points):
            for k in range(i, total_points):
                if k in elimination_right:
                    continue
                if self._data[self.y_column][i] < self._data[self.y_column][k]:
                    elimination_right.add(i)
                    continue
        elimination = elimination_left.intersection(elimination_right)
        self.results = self._data.drop(list(elimination))

    def save(self):
        new_name = self.filename.replace(".csv", "")
        new_name += "_EF.csv"
        self.results.to_csv(f"{new_name}")


dataset: Data


def main(argv):
    global dataset
    filename = argv[0]
    x = argv[1]
    y = argv[2]
    try:
        sheet = argv[3]
        dataset = Data(filename, x, y, sheet)
    except IndexError:
        dataset = Data(filename, x, y)
    finally:
        dataset.eliminate()
        dataset.save()


if __name__ == "__main__":
    main(sys.argv[1:])
