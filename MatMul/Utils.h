#pragma once

#include "Matrix.h"

HostMatrix generate(size_t Height, size_t Width);
HostMatrix referenceMul(HostMatrix &A, HostMatrix &B);