// ------------------------------
// <START> SETUP FOR POWERSORT
// ------------------------------

#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>

typedef long long npy_intp;

#define NPY_UNLIKELY(x) (x)
#define NPY_ENOMEM 1

struct IntTag {
    using type = long long;
    static bool less(const type &a, const type &b) {
        return a < b;
    }
};

// ------------------------------
// <END> SETUP FOR POWERSORT
// ------------------------------



/* -*- c -*- */

/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexicographic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

/* For details of Timsort, refer to
 * https://github.com/python/cpython/blob/3.7/Objects/listsort.txt
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION


#include <cstdlib>
#include <utility>

/* enough for 32 * 1.618 ** 128 elements.
   If powersort was used in all cases, 90 would suffice, as  32 * 2 ** 90  >=  32 * 1.618 ** 128  */
#define RUN_STACK_SIZE 128

static npy_intp
compute_min_run(npy_intp num)
{
    npy_intp r = 0;

    while (64 < num) {
        r |= num & 1;
        num >>= 1;
    }

    return num + r;
}

typedef struct {
    npy_intp s; /* start pointer */
    npy_intp l; /* length */
    int power;  /* node "level" for powersort merge strategy */
} run;

/* buffer for argsort. Declared here to avoid multiple declarations. */
typedef struct {
    npy_intp *pw;
    npy_intp size;
} buffer_intp;

/* buffer method */
static inline int
resize_buffer_intp(buffer_intp *buffer, npy_intp new_size)
{
    if (new_size <= buffer->size) {
        return 0;
    }

    npy_intp *new_pw = (npy_intp *)realloc(buffer->pw, new_size * sizeof(npy_intp));

    buffer->size = new_size;

    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
        return 0;
    }
}

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag>
struct buffer_ {
    typename Tag::type *pw;
    npy_intp size;
};

template <typename Tag>
static inline int
resize_buffer_(buffer_<Tag> *buffer, npy_intp new_size)
{
    using type = typename Tag::type;
    if (new_size <= buffer->size) {
        return 0;
    }

    type *new_pw = (type *)realloc(buffer->pw, new_size * sizeof(type));
    buffer->size = new_size;

    if (NPY_UNLIKELY(new_pw == NULL)) {
        return -NPY_ENOMEM;
    }
    else {
        buffer->pw = new_pw;
        return 0;
    }
}

template <typename Tag, typename type>
static npy_intp
count_run_(type *arr, npy_intp l, npy_intp num, npy_intp minrun)
{
    npy_intp sz;
    type vc, *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = arr + l;

    /* (not strictly) ascending sequence */
    if (!Tag::less(*(pl + 1), *pl)) {
        for (pi = pl + 1; pi < arr + num - 1 && !Tag::less(*(pi + 1), *pi);
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1; pi < arr + num - 1 && Tag::less(*(pi + 1), *pi);
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vc = *pi;
            pj = pi;

            while (pl < pj && Tag::less(vc, *(pj - 1))) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vc;
        }
    }

    return sz;
}

/* when the left part of the array (p1) is smaller, copy p1 to buffer
 * and merge from left to right
 */
template <typename Tag, typename type>
static void
merge_left_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3)
{
    type *end = p2 + l2;
    memcpy(p3, p1, sizeof(type) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        if (Tag::less(*p2, *p3)) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(type) * (p2 - p1));
    }
}

/* when the right part of the array (p2) is smaller, copy p2 to buffer
 * and merge from right to left
 */
template <typename Tag, typename type>
static void
merge_right_(type *p1, npy_intp l1, type *p2, npy_intp l2, type *p3)
{
    npy_intp ofs;
    type *start = p1 - 1;
    memcpy(p3, p2, sizeof(type) * l2);
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        if (Tag::less(*p3, *p1)) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(type) * ofs);
    }
}

/* Note: the naming convention of gallop functions are different from that of
 * CPython. For example, here gallop_right means gallop from left toward right,
 * whereas in CPython gallop_right means gallop
 * and find the right most element among equal elements
 */
template <typename Tag, typename type>
static npy_intp
gallop_right_(const type *arr, const npy_intp size, const type key)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr[0])) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr[ofs])) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[last_ofs] <= key < arr[ofs] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (Tag::less(key, arr[m])) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[ofs-1] <= key < arr[ofs] */
    return ofs;
}

template <typename Tag, typename type>
static npy_intp
gallop_left_(const type *arr, const npy_intp size, const type key)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (Tag::less(arr[size - 1], key)) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr[size - ofs - 1], key)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[size-ofs-1] < key <= arr[size-last_ofs-1] */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr[m], key)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[r-1] < key <= arr[r] */
    return r;
}

template <typename Tag, typename type>
static int
merge_at_(type *arr, const run *stack, const npy_intp at, buffer_<Tag> *buffer)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    type *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* arr[s2] belongs to arr[s1+k].
     * if try to comment this out for debugging purpose, remember
     * in the merging process the first element is skipped
     */
    k = gallop_right_<Tag>(arr + s1, l1, arr[s2]);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = arr + s1 + k;
    l1 -= k;
    p2 = arr + s2;
    /* arr[s2-1] belongs to arr[s2+l2] */
    l2 = gallop_left_<Tag>(arr + s2, l2, arr[s2 - 1]);

    if (l2 < l1) {
        ret = resize_buffer_<Tag>(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_right_<Tag>(p1, l1, p2, l2, buffer->pw);
    }
    else {
        ret = resize_buffer_<Tag>(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        merge_left_<Tag>(p1, l1, p2, l2, buffer->pw);
    }

    return 0;
}

/* See https://github.com/python/cpython/blob/ea23c897cd25702e72a04e06664f6864f07a7c5d/Objects/listsort.txt
*  for a detailed explanation.
*  In CPython, *num* is called *n*, but we changed it for consistency with the NumPy implementation.
*/
static int
powerloop(npy_intp s1, npy_intp n1, npy_intp n2, npy_intp num)
{
    int result = 0;
    npy_intp a = 2 * s1 + n1;  /* 2*a */
    npy_intp b = a + n1 + n2;  /* 2*b */
    for (;;) {
        ++result;
        if (a >= num) {  /* both quotient bits are 1 */
            a -= num;
            b -= num;
        }
        else if (b >= num) {  /* a/num bit is 0, b/num bit is 1 */
            break;
        }
        a <<= 1;
        b <<= 1;
    }
    return result;
}

template <typename Tag, typename type>
static int
found_new_run_(type *arr, run *stack, npy_intp *stack_ptr, npy_intp n2,
               npy_intp num, buffer_<Tag> *buffer)
{
    int ret;
    if (*stack_ptr > 0) {
        npy_intp s1 = stack[*stack_ptr - 1].s;
        npy_intp n1 = stack[*stack_ptr - 1].l;
        int power = powerloop(s1, n1, n2, num);
        while (*stack_ptr > 1 && stack[*stack_ptr - 2].power > power) {
            ret = merge_at_<Tag>(arr, stack, *stack_ptr - 2, buffer);
            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }
            stack[*stack_ptr - 2].l += stack[*stack_ptr - 1].l;
            --(*stack_ptr);
        }
        stack[*stack_ptr - 1].power = power;
    }
    return 0;
}

template <typename Tag, typename type>
static int
force_collapse_(type *arr, run *stack, npy_intp *stack_ptr,
                buffer_<Tag> *buffer)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = merge_at_<Tag>(arr, stack, top - 3, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

template <typename Tag>
static int
powersort_(void *start, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_<Tag> buffer;
    run stack[RUN_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);

    for (l = 0; l < num;) {
        n = count_run_<Tag>((type *)start, l, num, minrun);
        ret = found_new_run_<Tag>((type *)start, stack, &stack_ptr, n, num, &buffer);
        if (NPY_UNLIKELY(ret < 0))
            goto cleanup;

        // Push the new run onto the stack.
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        l += n;
    }

    ret = force_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;
cleanup:

    free(buffer.pw);

    return ret;
}

template <typename Tag, typename type>
static int
try_collapse_(type *arr, run *stack, npy_intp *stack_ptr, buffer_<Tag> *buffer)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = merge_at_<Tag>(arr, stack, top - 3, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
            }
        else if (1 < top && B <= C) {
            ret = merge_at_<Tag>(arr, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}


template <typename Tag>
static int
timsort_(void *start, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_<Tag> buffer;
    run stack[RUN_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);

    for (l = 0; l < num;) {
        n = count_run_<Tag>((type *)start, l, num, minrun);
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = try_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = force_collapse_<Tag>((type *)start, stack, &stack_ptr, &buffer);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;
    cleanup:

        free(buffer.pw);

    return ret;
}


/* argsort */

template <typename Tag, typename type>
static npy_intp
acount_run_(type *arr, npy_intp *tosort, npy_intp l, npy_intp num,
            npy_intp minrun)
{
    npy_intp sz;
    type vc;
    npy_intp vi;
    npy_intp *pl, *pi, *pj, *pr;

    if (NPY_UNLIKELY(num - l == 1)) {
        return 1;
    }

    pl = tosort + l;

    /* (not strictly) ascending sequence */
    if (!Tag::less(arr[*(pl + 1)], arr[*pl])) {
        for (pi = pl + 1;
             pi < tosort + num - 1 && !Tag::less(arr[*(pi + 1)], arr[*pi]);
             ++pi) {
        }
    }
    else { /* (strictly) descending sequence */
        for (pi = pl + 1;
             pi < tosort + num - 1 && Tag::less(arr[*(pi + 1)], arr[*pi]);
             ++pi) {
        }

        for (pj = pl, pr = pi; pj < pr; ++pj, --pr) {
            std::swap(*pj, *pr);
        }
    }

    ++pi;
    sz = pi - pl;

    if (sz < minrun) {
        if (l + minrun < num) {
            sz = minrun;
        }
        else {
            sz = num - l;
        }

        pr = pl + sz;

        /* insertion sort */
        for (; pi < pr; ++pi) {
            vi = *pi;
            vc = arr[*pi];
            pj = pi;

            while (pl < pj && Tag::less(vc, arr[*(pj - 1)])) {
                *pj = *(pj - 1);
                --pj;
            }

            *pj = vi;
        }
    }

    return sz;
}

template <typename Tag, typename type>
static npy_intp
agallop_right_(const type *arr, const npy_intp *tosort, const npy_intp size,
               const type key)
{
    npy_intp last_ofs, ofs, m;

    if (Tag::less(key, arr[tosort[0]])) {
        return 0;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size; /* arr[ofs] is never accessed */
            break;
        }

        if (Tag::less(key, arr[tosort[ofs]])) {
            break;
        }
        else {
            last_ofs = ofs;
            /* ofs = 1, 3, 7, 15... */
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[last_ofs]] <= key < arr[tosort[ofs]] */
    while (last_ofs + 1 < ofs) {
        m = last_ofs + ((ofs - last_ofs) >> 1);

        if (Tag::less(key, arr[tosort[m]])) {
            ofs = m;
        }
        else {
            last_ofs = m;
        }
    }

    /* now that arr[tosort[ofs-1]] <= key < arr[tosort[ofs]] */
    return ofs;
}

template <typename Tag, typename type>
static npy_intp
agallop_left_(const type *arr, const npy_intp *tosort, const npy_intp size,
              const type key)
{
    npy_intp last_ofs, ofs, l, m, r;

    if (Tag::less(arr[tosort[size - 1]], key)) {
        return size;
    }

    last_ofs = 0;
    ofs = 1;

    for (;;) {
        if (size <= ofs || ofs < 0) {
            ofs = size;
            break;
        }

        if (Tag::less(arr[tosort[size - ofs - 1]], key)) {
            break;
        }
        else {
            last_ofs = ofs;
            ofs = (ofs << 1) + 1;
        }
    }

    /* now that arr[tosort[size-ofs-1]] < key <= arr[tosort[size-last_ofs-1]]
     */
    l = size - ofs - 1;
    r = size - last_ofs - 1;

    while (l + 1 < r) {
        m = l + ((r - l) >> 1);

        if (Tag::less(arr[tosort[m]], key)) {
            l = m;
        }
        else {
            r = m;
        }
    }

    /* now that arr[tosort[r-1]] < key <= arr[tosort[r]] */
    return r;
}

template <typename Tag, typename type>
static void
amerge_left_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
             npy_intp *p3)
{
    npy_intp *end = p2 + l2;
    memcpy(p3, p1, sizeof(npy_intp) * l1);
    /* first element must be in p2 otherwise skipped in the caller */
    *p1++ = *p2++;

    while (p1 < p2 && p2 < end) {
        if (Tag::less(arr[*p2], arr[*p3])) {
            *p1++ = *p2++;
        }
        else {
            *p1++ = *p3++;
        }
    }

    if (p1 != p2) {
        memcpy(p1, p3, sizeof(npy_intp) * (p2 - p1));
    }
}

template <typename Tag, typename type>
static void
amerge_right_(type *arr, npy_intp *p1, npy_intp l1, npy_intp *p2, npy_intp l2,
              npy_intp *p3)
{
    npy_intp ofs;
    npy_intp *start = p1 - 1;
    memcpy(p3, p2, sizeof(npy_intp) * l2);
    p1 += l1 - 1;
    p2 += l2 - 1;
    p3 += l2 - 1;
    /* first element must be in p1 otherwise skipped in the caller */
    *p2-- = *p1--;

    while (p1 < p2 && start < p1) {
        if (Tag::less(arr[*p3], arr[*p1])) {
            *p2-- = *p1--;
        }
        else {
            *p2-- = *p3--;
        }
    }

    if (p1 != p2) {
        ofs = p2 - start;
        memcpy(start + 1, p3 - ofs + 1, sizeof(npy_intp) * ofs);
    }
}

template <typename Tag, typename type>
static int
amerge_at_(type *arr, npy_intp *tosort, const run *stack, const npy_intp at,
           buffer_intp *buffer)
{
    int ret;
    npy_intp s1, l1, s2, l2, k;
    npy_intp *p1, *p2;
    s1 = stack[at].s;
    l1 = stack[at].l;
    s2 = stack[at + 1].s;
    l2 = stack[at + 1].l;
    /* tosort[s2] belongs to tosort[s1+k] */
    k = agallop_right_<Tag>(arr, tosort + s1, l1, arr[tosort[s2]]);

    if (l1 == k) {
        /* already sorted */
        return 0;
    }

    p1 = tosort + s1 + k;
    l1 -= k;
    p2 = tosort + s2;
    /* tosort[s2-1] belongs to tosort[s2+l2] */
    l2 = agallop_left_<Tag>(arr, tosort + s2, l2, arr[tosort[s2 - 1]]);

    if (l2 < l1) {
        ret = resize_buffer_intp(buffer, l2);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_right_<Tag>(arr, p1, l1, p2, l2, buffer->pw);
    }
    else {
        ret = resize_buffer_intp(buffer, l1);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }

        amerge_left_<Tag>(arr, p1, l1, p2, l2, buffer->pw);
    }

    return 0;
}

template <typename Tag, typename type>
static int
atry_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
               buffer_intp *buffer)
{
    int ret;
    npy_intp A, B, C, top;
    top = *stack_ptr;

    while (1 < top) {
        B = stack[top - 2].l;
        C = stack[top - 1].l;

        if ((2 < top && stack[top - 3].l <= B + C) ||
            (3 < top && stack[top - 4].l <= stack[top - 3].l + B)) {
            A = stack[top - 3].l;

            if (A <= C) {
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 3].l += B;
                stack[top - 2] = stack[top - 1];
                --top;
            }
            else {
                ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

                if (NPY_UNLIKELY(ret < 0)) {
                    return ret;
                }

                stack[top - 2].l += C;
                --top;
            }
        }
        else if (1 < top && B <= C) {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += C;
            --top;
        }
        else {
            break;
        }
    }

    *stack_ptr = top;
    return 0;
}

template <typename Tag, typename type>
static int
aforce_collapse_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                 buffer_intp *buffer)
{
    int ret;
    npy_intp top = *stack_ptr;

    while (2 < top) {
        if (stack[top - 3].l <= stack[top - 1].l) {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 3, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 3].l += stack[top - 2].l;
            stack[top - 2] = stack[top - 1];
            --top;
        }
        else {
            ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

            if (NPY_UNLIKELY(ret < 0)) {
                return ret;
            }

            stack[top - 2].l += stack[top - 1].l;
            --top;
        }
    }

    if (1 < top) {
        ret = amerge_at_<Tag>(arr, tosort, stack, top - 2, buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            return ret;
        }
    }

    return 0;
}

template <typename Tag, typename type>
static int
apower_found_new_run_(type *arr, npy_intp *tosort, run *stack, npy_intp *stack_ptr,
                      npy_intp n2, npy_intp num, buffer_intp *buffer)
{
    // identical shape to found_new_run_ but works on amerge_*/agallop_* via amerge_at_
    int ret;
    if (*stack_ptr > 0) {
        npy_intp s1 = stack[*stack_ptr - 1].s;
        npy_intp n1 = stack[*stack_ptr - 1].l;
        int power = powerloop(s1, n1, n2, num);
        while (*stack_ptr > 1 && stack[*stack_ptr - 2].power > power) {
            ret = amerge_at_<Tag>(arr, tosort, stack, *stack_ptr - 2, buffer);
            if (NPY_UNLIKELY(ret < 0)) return ret;
            stack[*stack_ptr - 2].l += stack[*stack_ptr - 1].l;
            --(*stack_ptr);
        }
        stack[*stack_ptr - 1].power = power;
    }
    return 0;
}

template <typename Tag>
static int
apowersort_(void *v, npy_intp *tosort, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l = 0, n, stack_ptr = 0, minrun = compute_min_run(num);
    buffer_intp buffer{nullptr, 0};
    run stack[RUN_STACK_SIZE];

    for (; l < num; ) {
        n = acount_run_<Tag>((type*)v, tosort, l, num, minrun);
        ret = apower_found_new_run_<Tag>((type*)v, tosort, stack, &stack_ptr, n, num, &buffer);
        if (NPY_UNLIKELY(ret < 0)) goto cleanup;

        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        l += n;
    }

    ret = aforce_collapse_<Tag>((type*)v, tosort, stack, &stack_ptr, &buffer);
    if (NPY_UNLIKELY(ret < 0)) goto cleanup;

    ret = 0;
    cleanup:
        if (buffer.pw) free(buffer.pw);
    return ret;
}


template <typename Tag>
static int
atimsort_(void *v, npy_intp *tosort, npy_intp num)
{
    using type = typename Tag::type;
    int ret;
    npy_intp l, n, stack_ptr, minrun;
    buffer_intp buffer;
    run stack[RUN_STACK_SIZE];
    buffer.pw = NULL;
    buffer.size = 0;
    stack_ptr = 0;
    minrun = compute_min_run(num);

    for (l = 0; l < num;) {
        n = acount_run_<Tag>((type *)v, tosort, l, num, minrun);
        stack[stack_ptr].s = l;
        stack[stack_ptr].l = n;
        ++stack_ptr;
        ret = atry_collapse_<Tag>((type *)v, tosort, stack, &stack_ptr,
                                  &buffer);

        if (NPY_UNLIKELY(ret < 0)) {
            goto cleanup;
        }

        l += n;
    }

    ret = aforce_collapse_<Tag>((type *)v, tosort, stack, &stack_ptr, &buffer);

    if (NPY_UNLIKELY(ret < 0)) {
        goto cleanup;
    }

    ret = 0;
cleanup:

    if (buffer.pw != NULL) {
        free(buffer.pw);
    }

    return ret;
}

namespace testing_utils {
    inline std::string vec_to_string(const std::vector<long long>& v) {
        constexpr size_t max_elems = 20;
        std::ostringstream oss;

        oss << "[";
        const size_t n = v.size();
        for (size_t i = 0; i < n && i < max_elems; ++i) {
            if (i > 0) oss << ", ";
            oss << v[i];
        }
        if (n > max_elems) oss << ", ...";
        oss << "]";

        return oss.str();
    }

    std::vector<long long> list_from_file(const std::string &filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            return std::vector<long long>();
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        size_t start = content.find('[');
        size_t end = content.rfind(']');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            std::cerr << "Error: Invalid format in file " << filename << std::endl;
            return std::vector<long long>();
        }
        std::string list_str = content.substr(start + 1, end - start - 1);

        std::vector<long long> result;
        std::stringstream ss(list_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Remove all whitespace characters from the token
            token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
            if (!token.empty()) {
                try {
                    long long value = std::stoll(token);
                    result.push_back(value);
                } catch (const std::exception &e) {
                    std::cerr << "Error parsing token: " << token << " in file " << filename << std::endl;
                    return std::vector<long long>();
                }
            }
        }
        return result;
    }
}


int main() {
    const std::string folder_path = "submissions";
    constexpr int repetitions = 10;

    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        std::string filename = entry.path().string();

        std::vector<long long> base_data = testing_utils::list_from_file(filename);
        const std::size_t num_elements = base_data.size();

        std::cout << "\n" << "File: " << filename << "\n"
                          << "Length: " << num_elements << "\n";

        auto benchmark = [&](auto sort_func) {
            std::vector<double> durations;
            durations.reserve(repetitions);
            for (int rep = 0; rep < repetitions; ++rep) {
                auto data = base_data; // copy each run
                auto t_start = std::chrono::steady_clock::now();
                sort_func(static_cast<void*>(data.data()), static_cast<npy_intp>(data.size()));
                auto t_end = std::chrono::steady_clock::now();
                durations.push_back(std::chrono::duration<double, std::micro>(t_end - t_start).count());
            }
            return std::accumulate(durations.begin(), durations.end(), 0.0) /durations.size();
        };

        // Reuse single benchmark() by wrapping argsort calls.
        // Preallocate index buffer once per file to reduce alloc overhead.
        std::vector<npy_intp> idx(base_data.size());

        auto benchmark_argsort_tim = [&] {
            return benchmark([&](void* ptr, npy_intp n){
                std::iota(idx.begin(), idx.begin() + n, static_cast<npy_intp>(0));
                (void)atimsort_<IntTag>(ptr, idx.data(), n);
            });
        };

        auto benchmark_argsort_power = [&](){
            return benchmark([&](void* ptr, npy_intp n){
                std::iota(idx.begin(), idx.begin() + n, static_cast<npy_intp>(0));
                (void)apowersort_<IntTag>(ptr, idx.data(), n);
            });
        };

        double power_mean = benchmark([](void* ptr, npy_intp n) {powersort_<IntTag>(ptr, n);});
        double tim_mean = benchmark([](void* ptr, npy_intp n) {timsort_<IntTag>(ptr, n);});
        double atim_mean  = benchmark_argsort_tim();
        double apower_mean= benchmark_argsort_power();

        std::cout << "  PowerSort (in-place) mean [us]: " << power_mean  << "\n"
                  << "  TimSort  (in-place)  mean [us]: " << tim_mean    << "\n"
                  << "  ArgSort TimSort      mean [us]: " << atim_mean   << "\n"
                  << "  ArgSort PowerSort    mean [us]: " << apower_mean << "\n";
    }

    return 0;
}