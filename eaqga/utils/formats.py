# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math


def formatMantissa(value: float, decimals: int = 1) -> str:

    """
    Formata um float em notação de mantissa base 10.

    Exemplo:
        0.0000036 -> "3.6 × 10^-6"
        1.5e7     -> "1.5 × 10^7"
        0.0       -> "0"
    """

    if value == 0.0:
        return "0"

    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)

    return f"{mantissa:.{decimals}f} × 10^{exponent}"


def formatValue(
    value: float,
    decimals: int = 2,
    *,
    simple_limit: float = 100.0,
) -> str:
    """Formata valores simples com casas fixas e grandes em mantissa.

    Para ``abs(value) <= simple_limit``, usa ``f"{value:.{decimals}f}"``.
    Acima disso, delega para :func:`formatMantissa`.

    Valores muito pequenos (por exemplo ``0.001``) aparecem como ``0.00``
    com duas casas. Para precisão científica nesses casos, use
    :func:`formatMantissa` diretamente.
    """

    if not math.isfinite(value):
        return str(value)

    if abs(value) <= simple_limit:
        return f"{value:.{decimals}f}"

    return formatMantissa(value)


def formatLargeNumber(value: int | float, decimals: int = 1) -> str:

    '''
    Formata um número longo para uma representação compacta com sufixo.

    Detecta a classe de tamanho do número e aplica o sufixo correspondente:
        → Bilhões:   B
        → Milhões:   M
        → Milhares:  K
        → Unidades:  sem sufixo

    @params:
        → value:    o número a ser formatado.          mandatory!
        → decimals: casas decimais no resultado.        default: 1
    '''

    absValue = abs(value)

    if absValue >= 1_000_000_000:
        formatted = f"{absValue / 1_000_000_000:.{decimals}f}B"
    elif absValue >= 1_000_000:
        formatted = f"{absValue / 1_000_000:.{decimals}f}M"
    elif absValue >= 1_000:
        formatted = f"{absValue / 1_000:.{decimals}f}K"
    else:
        formatted = f"{absValue:.{decimals}f}"

    if value < 0:
        formatted = "-" + formatted

    return formatted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
