colors = {
    "red": (255, 0, 0,255),
    "green": (0, 255, 0,255),
    "blue": (0, 0, 255,255),
    "white": (255, 255, 255,255),
    "black": (0, 0, 0,255),
}


class Color:
    def __init__(self) -> None:
        self.index = -1

    def __call__(self, name):
        return self.get_color(name)

    def next_color(self):
        index = (index + 1) % len(colors)
        color = colors.values()[index]
        return color

    def get_color(self, name):
        if name not in colors:
            raise Exception(f"Color {name} does not exist")
        return colors[name]

    def current_color(self):
        return colors.values()[self.index]

    def current_name_color(self):
        return colors.keys()[self.index]
