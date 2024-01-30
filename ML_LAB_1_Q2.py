def range(elements):
    if len(elements) < 3:
        return "Range determination not possible"

    elements.sort()
    return elements[-1] - elements[0]
#intializing the list
elements = [5, 3, 8, 1, 0, 4]
range_value = range(elements)

if range_value is not None:
    print(f"The range is {range_value}")
else:
    print(range_value)