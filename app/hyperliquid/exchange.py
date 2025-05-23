class Exchange:
    def __init__(self, wallet):
        self.wallet = wallet

    def order(self, coin: str, is_buy: bool, sz: float, limit_px=None, reduce_only=False):
        side = "buy" if is_buy else "sell"
        order_payload = {
            "coin": coin,
            "is_buy": is_buy,
            "sz": sz,
            "limit_px": limit_px,
            "reduce_only": reduce_only,
        }
        return self.wallet.sign_and_send("order", order_payload)
