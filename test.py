from termcolor import colored
import pyfiglet

ascii_art = pyfiglet.figlet_format("Hello, World!")
colored_ascii = colored(ascii_art, 'cyan')

print(colored_ascii)