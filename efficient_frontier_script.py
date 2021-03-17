import pandas
import sys


class Data:
    _data: pandas.DataFrame
    results: pandas.DataFrame = None
    filename: str = None
    x_column: str = None
    y_column: str = None
    factor_x = 1
    factor_y = 1

    def __init__(self, filename, x, y, direction, sheet=None):
        self.x_column = x
        self.y_column = y
        self.filename = filename
        if ".xlsx" in filename:
            excel_file = pandas.ExcelFile(filename)
            self._data = pandas.read_excel(excel_file, sheet)
        elif ".csv" in filename:
            self._data = pandas.read_csv(filename)

        if direction == "MinMax":
            self.factor_x = -1
        elif direction == "MaxMin":
            self.factor_y = -1
        elif direction == "MinMin":
            self.factor_x = -1
            self.factor_y = -1

        self._data = self._data.sort_values(by=x).reset_index()

    def eliminate(self):
        total_points = self._data[self.x_column].size
        elimination = set()
        for i in range(total_points):
            if i in elimination:
                continue
            for k in range(total_points):
                if k == i:
                    continue
                # if k in elimination:
                #     continue
                if self.factor_x * self._data[self.x_column][i] > self.factor_x * self._data[self.x_column][k]\
                        and self.factor_y * self._data[self.y_column][i] > self.factor_y * self._data[self.y_column][k]:
                    elimination.add(k)
                elif self.factor_x * self._data[self.x_column][i] < self.factor_x * self._data[self.x_column][k] \
                        and self.factor_y * self._data[self.y_column][i] < self.factor_y * self._data[self.y_column][k]:
                    elimination.add(i)
                    break
        # elimination_right = set()
        # for i in range(total_points):
        #     for k in range(i, total_points):
        #         if k in elimination_right:
        #             continue
        #         if self._data[self.y_column][i] < self._data[self.y_column][k]:
        #             elimination_right.add(i)
        #             continue
        # elimination = elimination_left.intersection(elimination_right)
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
    direction = argv[3]
    try:
        sheet = argv[4]
        dataset = Data(filename, x, y, direction, sheet)
    except IndexError:
        dataset = Data(filename, x, y, direction)
    finally:
        dataset.eliminate()
        dataset.save()


if __name__ == "__main__":
    main(sys.argv[1:])
