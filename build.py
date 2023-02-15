import os
import subprocess

def main():
    os.makedirs('build', exist_ok=True)
    subprocess.check_call('cmake -G "Ninja" -B build -DCMAKE_CXX_STANDARD=20', shell=True)

if __name__ == '__main__':
    main()
