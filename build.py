import os
import subprocess

def main():
    os.makedirs('build', exist_ok=True)
    subprocess.check_call('cmake -G "Ninja" -B build -DCMAKE_CXX_STANDARD=20 -DCMAKE_TOOLCHAIN_FILE=/home/jp/Documents/projects/C++/vcpkg/scripts/buildsystems/vcpkg.cmake', shell=True)

if __name__ == '__main__':
    main()
