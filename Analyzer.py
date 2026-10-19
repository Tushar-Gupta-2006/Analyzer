import numpy as np


class DataAnalytics:

    def __init__(self, data=None):
        self._array = np.array(data) if data is not None else None

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        self._array = np.array(value)

    # Encapsulated private helper methods for internal computations
    def _validate_array(self):
        if self._array is None:
            raise ValueError(
                "No array loaded. Please create or load an array first."
            )

    @staticmethod
    def parse_input_to_array(elements_str, shape):
        """Static utility method to convert user input string into a shaped NumPy array."""
        data = list(map(float, elements_str.strip().split()))
        expected_size = np.prod(shape)
        if len(data) != expected_size:
            raise ValueError(
                f"Expected {expected_size} elements, but got {len(data)}."
            )
        return np.array(data).reshape(shape)

    @classmethod
    def create_zeros(cls, shape):
        """Class method utility to construct a DataAnalytics object with zero array."""
        return cls(np.zeros(shape))

    # Array Management Methods
    def create_array(self, dim, shape, elements_str):
        self._array = self.parse_input_to_array(elements_str, shape)
        return self._array

    def slice_array(self, row_slice=None, col_slice=None):
        self._validate_array()
        if self._array.ndim == 1:
            return self._array[row_slice]
        elif self._array.ndim == 2:
            return self._array[row_slice, col_slice]
        else:
            return self._array[row_slice]

    def combine_array(self, other_array, axis=0):
        self._validate_array()
        return np.vstack((self._array, other_array))

    def split_array(self, sections, axis=0):
        self._validate_array()
        return np.array_split(self._array, sections, axis=axis)

    # Mathematical Operations
    def add(self, other_array):
        self._validate_array()
        return np.add(self._array, other_array)

    def subtract(self, other_array):
        self._validate_array()
        return np.subtract(self._array, other_array)

    def multiply(self, other_array):
        self._validate_array()
        return np.multiply(self._array, other_array)

    def divide(self, other_array):
        self._validate_array()
        return np.divide(self._array, other_array)

    def dot_product(self, other_array):
        self._validate_array()
        return np.dot(self._array, other_array)

    # Search, Sort, and Filter
    def search_value(self, value):
        self._validate_array()
        return np.argwhere(self._array == value)

    def sort_array(self, axis=-1):
        self._validate_array()
        return np.sort(self._array, axis=axis)

    def filter_values(self, condition_str):
        self._validate_array()
        # Safely evaluate simple conditions like "> 20"
        mask = eval(f"self._array {condition_str}")
        return self._array[mask]

    # Aggregating and Statistical Functions
    def get_sum(self):
        self._validate_array()
        return np.sum(self._array)

    def get_mean(self):
        self._validate_array()
        return np.mean(self._array)

    def get_median(self):
        self._validate_array()
        return np.median(self._array)

    def get_std(self):
        self._validate_array()
        return np.std(self._array)

    def get_var(self):
        self._validate_array()
        return np.var(self._array)

    def get_min_max(self):
        self._validate_array()
        return np.min(self._array), np.max(self._array)

    def get_percentile(self, q):
        self._validate_array()
        return np.percentile(self._array, q)

    @staticmethod
    def correlation(arr1, arr2):
        return np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]


def parse_slice(slice_str):
    parts = list(map(int, slice_str.split(":")))
    return slice(parts[0], parts[1])


def main():
    analytics = DataAnalytics()

    print("Welcome to the NumPy Analyzer!")
    print("==============================================")

    while True:
        print("\nChoose an option:")
        print("1. Create a Numpy Array")
        print("2. Perform Mathematical Operations")
        print("3. Combine or Split Arrays")
        print("4. Search, Sort, or Filter Arrays")
        print("5. Compute Aggregates and Statistics")
        print("6. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            print("\nSelect the type of array to create:")
            print("1. 1D Array")
            print("2. 2D Array")
            print("3. 3D Array")
            dim_choice = input("Enter your choice: ").strip()

            if dim_choice == "1":
                size = int(input("Enter length of array: "))
                shape = (size,)
            elif dim_choice == "2":
                r = int(input("Enter the number of rows: "))
                c = int(input("Enter the number of columns: "))
                shape = (r, c)
            elif dim_choice == "3":
                d = int(input("Enter depth: "))
                r = int(input("Enter rows: "))
                c = int(input("Enter columns: "))
                shape = (d, r, c)
            else:
                print("Invalid choice!")
                continue

            elem_str = input(
                f"Enter {np.prod(shape)} elements for the array separated by space: "
            )
            try:
                arr = analytics.create_array(int(dim_choice), shape, elem_str)
                print("\nArray created successfully:")
                print(arr)

                # Sub-menu for Slicing/Indexing right after creation
                print("\nChoose an operation:")
                print("1. Indexing")
                print("2. Slicing")
                print("3. Go Back")
                sub_choice = input("Enter your choice: ").strip()

                if sub_choice == "2" and arr.ndim == 2:
                    r_slice = input(
                        "Enter the row range (start:end): "
                    ).strip()
                    c_slice = input(
                        "Enter the column range (start:end): "
                    ).strip()
                    sliced = analytics.slice_array(
                        parse_slice(r_slice), parse_slice(c_slice)
                    )
                    print("\nSliced Array:")
                    print(sliced)
            except Exception as e:
                print(f"Error: {e}")

        elif choice == "2":
            if analytics.array is None:
                print("Please create an array first!")
                continue
            print("\nChoose a mathematical operation:")
            print("1. Addition")
            print("2. Subtraction")
            print("3. Multiplication")
            print("4. Division")
            math_choice = input("Enter your choice: ").strip()

            elem_str = input(
                f"\nEnter the same-size array elements ({analytics.array.size} elements separated by space): "
            )
            try:
                second_arr = DataAnalytics.parse_input_to_array(
                    elem_str, analytics.array.shape
                )
                print("\nOriginal Array:")
                print(analytics.array)
                print("\nSecond Array:")
                print(second_arr)

                if math_choice == "1":
                    res = analytics.add(second_arr)
                    print("\nResult of Addition:")
                elif math_choice == "2":
                    res = analytics.subtract(second_arr)
                    print("\nResult of Subtraction:")
                elif math_choice == "3":
                    res = analytics.multiply(second_arr)
                    print("\nResult of Multiplication:")
                elif math_choice == "4":
                    res = analytics.divide(second_arr)
                    print("\nResult of Division:")
                print(res)
            except Exception as e:
                print(f"Error: {e}")

        elif choice == "3":
            if analytics.array is None:
                print("Please create an array first!")
                continue
            print("\nChoose an option:")
            print("1. Combine Arrays")
            print("2. Split Array")
            cs_choice = input("Enter your choice: ").strip()

            if cs_choice == "1":
                elem_str = input(
                    f"\nEnter the elements of another array to combine ({analytics.array.size} elements separated by space): "
                )
                try:
                    second_arr = DataAnalytics.parse_input_to_array(
                        elem_str, analytics.array.shape
                    )
                    print("\nOriginal Array:")
                    print(analytics.array)
                    print("\nSecond Array:")
                    print(second_arr)
                    combined = analytics.combine_array(second_arr)
                    print("\nCombined Array (Vertical Stack):")
                    print(combined)
                except Exception as e:
                    print(f"Error: {e}")

        elif choice == "4":
            if analytics.array is None:
                print("Please create an array first!")
                continue
            print("\nChoose an option:")
            print("1. Search a value")
            print("2. Sort the array")
            print("3. Filter values")
            ssf_choice = input("Enter your choice: ").strip()

            if ssf_choice == "2":
                print("\nOriginal Array:")
                print(analytics.array)
                sorted_arr = analytics.sort_array(axis=-1)
                print("\nSorted Array:")
                print(sorted_arr)
                print("(Sorting applied row-wise.)")

        elif choice == "5":
            if analytics.array is None:
                print("Please create an array first!")
                continue
            print("\nChoose an aggregate/statistical operation:")
            print("1. Sum")
            print("2. Mean")
            print("3. Median")
            print("4. Standard Deviation")
            print("5. Variance")
            stat_choice = input("Enter your choice: ").strip()

            print("\nOriginal Array:")
            print(analytics.array)

            if stat_choice == "1":
                print(f"\nSum of Array: {analytics.get_sum()}")
            elif stat_choice == "2":
                print(f"\nMean of Array: {analytics.get_mean()}")
            elif stat_choice == "3":
                print(f"\nMedian of Array: {analytics.get_median()}")
            elif stat_choice == "4":
                print(
                    f"\nStandard Deviation of Array: {analytics.get_std()}"
                )
            elif stat_choice == "5":
                print(f"\nVariance of Array: {analytics.get_var()}")

        elif choice == "6":
            print("\nThank you for using the NumPy Analyzer! Goodbye!")
            break


if __name__ == "__main__":
    main()