#ifndef _LIB_DER_H_
#define _LIB_DER_H_

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *data;
    size_t length;
} DERItem;

#endif // _LIB_DER_H_
