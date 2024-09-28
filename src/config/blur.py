class BlurSteps:
    def __init__(self, steps) -> None:
        self.steps = steps
        self.index = -1

    def next(self):
        self.index += 1
        return self.steps[self.index]

    def restart(self):
        self.index = -1


class BlurFunctions:
    def lineal(length, strong=1):
        return BlurSteps(
            steps=[(1 - (x / length)) * strong for x in range(length)]
        )


class Blur:
    default_function = BlurFunctions.lineal
