#ifndef Glucose_Clone_h
#define Glucose_Clone_h


namespace Glucose {

    class Clone {
        public:
          virtual Clone* clone(int threadId) const = 0;
    };
};

#endif
