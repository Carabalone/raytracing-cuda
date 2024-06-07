#ifndef CLOCK_H
#define CLOCK_H

#include <chrono>
#include <iostream>

namespace rtweekend {

    class clock {
    private:
        std::chrono::time_point<std::chrono::system_clock> _start;
        std::chrono::time_point<std::chrono::system_clock> _end;
        std::chrono::duration<double> elapsed_sec;

    public:
        clock() : _start(), _end(), elapsed_sec(0) {}

        void start() {
            _start = std::chrono::system_clock::now();
        }
        void end() {
            _end = std::chrono::system_clock::now();
            elapsed_sec = _end - _start;
        }

        void print() {
            std::time_t end_time = std::chrono::system_clock::to_time_t(_end);
            std::cout << "Finished computation at " << std::ctime(&end_time)
                      << "Elapsed time: " << elapsed_sec.count() * 1000.0f << "ms\n";
        }
    };
}

#endif // CLOCK_H

