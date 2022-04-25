# -*- coding: utf-8 -*-

import collections as _collections
_collections_abc = getattr(_collections, 'abc', _collections)
import decimal as _decimal
import fractions as _fractions
import numbers as _numbers
import random as _random
import sys as _sys

_np = None
try:
    import numpy as _np
except ImportError:
    pass

Hashable = (_collections_abc.Hashable, )
Iterable = (_collections_abc.Iterable, )
Sequence = (_collections_abc.Sequence, )
Mapping = (_collections_abc.Mapping, )
String = [ str, bytes, bytearray, memoryview ]
if _sys.version_info.major < 3:
    String.insert(0, basestring)
    String.insert(2, unicode)
String = tuple(String)
Number = (_numbers.Number, )
Complex = (_numbers.Complex, complex)
Real = (_numbers.Real, )
Rational = (_numbers.Rational, _decimal.Decimal, _fractions.Fraction)
Floating = (float, )
Integer = (_numbers.Integral, int)
Boolean = (bool, )
RandomState = [ _random.Random ]
if (
    hasattr(_random, 'SystemRandom') and
    _random.SystemRandom is not None and
    isinstance(_random.SystemRandom, type)
):
    RandomState.append(_random.SystemRandom)
RandomState = tuple(RandomState)

NumpyHashable = None
NumpyIterable = None
NumpySequence = None
NumpyMapping = None
NumpyString = None
NumpyNumber = None
NumpyComplex = None
NumpyReal = None
NumpyRational = None
NumpyFloating = None
NumpyInteger = None
NumpyBoolean = None
NumpyBitGenerator = None
NumpyRandomState = None
if _np is not None:
    NumpyHashable = ()
    NumpyIterable = (_np.ndarray, )
    NumpySequence = (_np.ndarray, )
    NumpyMapping = ()
    NumpyString = [ _np.str_, _np.bytes_ ]
    if hasattr(_np, 'unicode_') and _np.unicode_ is not _np.str_:
        NumpyString.insert(1, _np.unicode_)
    NumpyString = tuple(NumpyString)
    NumpyNumber = (_np.number, )
    NumpyComplex = (_np.complexfloating, )
    NumpyReal = ()
    NumpyRational = ()
    NumpyFloating = (_np.floating, )
    NumpyInteger = (_np.integer, )
    NumpyBoolean = (_np.bool_, )
    NumpyRandomState = list()
    if (
        hasattr(_np.random, 'BitGenerator') and
        _np.random.BitGenerator is not None
    ):
        NumpyBitGenerator = (_np.random.BitGenerator, )
    if (
        hasattr(_np.random, 'RandomState') and
        _np.random.RandomState is not None
    ):
        NumpyRandomState.append(_np.random.RandomState)
    if (
        hasattr(_np.random, 'Generator') and
        _np.random.Generator is not None
    ):
        NumpyRandomState.append(_np.random.Generator)
    NumpyRandomState = tuple(NumpyRandomState)

AnyHashable = Hashable if NumpyHashable is None else Hashable + NumpyHashable
AnyIterable = Iterable if NumpyIterable is None else Iterable + NumpyIterable
AnySequence = Sequence if NumpySequence is None else Sequence + NumpySequence
AnyMapping = Mapping if NumpyMapping is None else Mapping + NumpyMapping
AnyString = String if NumpyString is None else String + NumpyString
AnyNumber = Number if NumpyNumber is None else Number + NumpyNumber
AnyComplex = Complex if NumpyComplex is None else Complex + NumpyComplex
AnyReal = Real if NumpyReal is None else Real + NumpyReal
AnyRational = Rational if NumpyRational is None else Rational + NumpyRational
AnyFloating = Floating if NumpyFloating is None else Floating + NumpyFloating
AnyInteger = Integer if NumpyInteger is None else Integer + NumpyInteger
AnyBoolean = Boolean if NumpyBoolean is None else Boolean + NumpyBoolean
AnyRandomState = \
    RandomState if NumpyRandomState is None \
        else RandomState + NumpyRandomState
