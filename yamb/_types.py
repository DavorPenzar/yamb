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
Collection = (_collections_abc.Collection, )
Sequence = (_collections_abc.Sequence, )
Mapping = (_collections_abc.Mapping, )
String = [ str, bytearray, memoryview ]
Bytes = [ bytearray, memoryview ]
if _sys.version_info.major < 3:
    String.insert(0, basestring)
    String.insert(2, unicode)
    Bytes.insert(0, str)
else:
    String.insert(1, bytes)
    Bytes.insert(0, bytes)
String = tuple(String)
Bytes = tuple(Bytes)
Number = (_numbers.Number, )
Complex = (_numbers.Complex, complex)
Real = (_numbers.Real, )
Rational = (_numbers.Rational, _fractions.Fraction, _decimal.Decimal)
Floating = (float, )
Integer = (_numbers.Integral, int)
Boolean = (bool, )
RandomState = [ _random.Random ]
if (hasattr(_random, 'SystemRandom') and _random.SystemRandom is not None):
    RandomState.append(_random.SystemRandom)
RandomState = tuple(RandomState)

NumpyHashable = None
NumpyIterable = None
NumpyCollection = None
NumpySequence = None
NumpyMapping = None
NumpyString = None
NumpyBytes = None
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
    NumpyCollection = (_np.ndarray, )
    NumpySequence = (_np.ndarray, )
    NumpyMapping = ()
    NumpyString = [ _np.str_ ]
    NumpyBytes = []
    if (
        hasattr(_np, 'unicode_') and
        not (_np.unicode_ is None or _np.unicode_ is _np.str_)
    ):
        NumpyString.insert(1, _np.unicode_)
    if (
        hasattr(_np, 'bytes_') and
        not (_np.bytes_ is None or _np.bytes_ is _np.str_)
    ):
        NumpyString.append(_np.bytes_)
        NumpyBytes.append(_np.bytes_)
    NumpyString = tuple(NumpyString)
    NumpyBytes = tuple(NumpyBytes)
    NumpyNumber = (_np.number, )
    NumpyComplex = (_np.complexfloating, )
    NumpyReal = ()
    NumpyRational = ()
    NumpyFloating = (_np.floating, )
    NumpyInteger = (_np.integer, )
    NumpyBoolean = (_np.bool_, )
    if (
        hasattr(_np.random, 'BitGenerator') and
        _np.random.BitGenerator is not None
    ):
        NumpyBitGenerator = (_np.random.BitGenerator, )
    else:
        NumpyBitGenerator = ()
    NumpyRandomState = list()
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
AnyCollection = \
    Collection if NumpyCollection is None else Collection + NumpyCollection
AnySequence = Sequence if NumpySequence is None else Sequence + NumpySequence
AnyMapping = Mapping if NumpyMapping is None else Mapping + NumpyMapping
AnyString = String if NumpyString is None else String + NumpyString
AnyBytes = Bytes if NumpyBytes is None else Bytes + NumpyBytes
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
