import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import logging
from collections import deque

class OrderBookAnalyzer:
    def __init__(self, symbol, depth=10, history_length=100):
        """
        Initializes the order book analyzer for market microstructure analysis.
        
        Args:
            symbol: Trading symbol (e.g., "GBPJPY")
            depth: Depth of the order book to analyze
            history_length: Number of snapshots to keep in history
        """
        self.symbol = symbol
        self.depth = depth
        self.history = deque(maxlen=history_length)
        self.imbalance_history = deque(maxlen=history_length)
        self.liquidity_history = deque(maxlen=history_length)
        self.last_update_time = None
        
        # Logger setup
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger('order_book_analyzer')
    
    def get_order_book(self):
        """
        Retrieves the current order book from MT5.
        
        Returns:
            dict: Order book data or None if not available
        """
        try:
            # Get order book from MT5
            book = mt5.market_book_get(self.symbol)
            
            if book is None or len(book) < 2:
                self.logger.warning(f"No order book data available for {self.symbol}")
                return None
            
            # Convert to more usable format
            bids = []
            asks = []
            
            for item in book:
                if item.type == mt5.BOOK_TYPE_SELL:
                    asks.append({"price": item.price, "volume": item.volume})
                elif item.type == mt5.BOOK_TYPE_BUY:
                    bids.append({"price": item.price, "volume": item.volume})
            
            # Sort bids (descending) and asks (ascending)
            bids.sort(key=lambda x: x["price"], reverse=True)
            asks.sort(key=lambda x: x["price"])
            
            # Limit to specified depth
            bids = bids[:self.depth]
            asks = asks[:self.depth]
            
            result = {
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now()
            }
            
            # Add to history
            self.history.append(result)
            self.last_update_time = result["timestamp"]
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error retrieving order book: {str(e)}")
            return None
    
    def calculate_imbalance(self, order_book=None):
        """
        Calculates order book imbalance metrics.
        
        Args:
            order_book: Order book data (optional, retrieves new data if None)
            
        Returns:
            dict: Imbalance metrics
        """
        if order_book is None:
            order_book = self.get_order_book()
        
        if order_book is None:
            return None
        
        bids = order_book["bids"]
        asks = order_book["asks"]
        
        # Calculate total volumes
        total_bid_volume = sum(bid["volume"] for bid in bids)
        total_ask_volume = sum(ask["volume"] for ask in asks)
        
        # Basic imbalance ratio
        if total_ask_volume > 0:
            imbalance_ratio = total_bid_volume / total_ask_volume
        else:
            imbalance_ratio = 999  # Very high if no asks
        
        # Weighted imbalance that gives more weight to prices near the spread
        weighted_bid_volume = 0
        weighted_ask_volume = 0
        
        if bids and asks:
            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]
            
            for i, bid in enumerate(bids):
                weight = 1 / (1 + i)  # Higher weight for closer to the spread
                weighted_bid_volume += bid["volume"] * weight
            
            for i, ask in enumerate(asks):
                weight = 1 / (1 + i)  # Higher weight for closer to the spread
                weighted_ask_volume += ask["volume"] * weight
            
            weighted_imbalance = weighted_bid_volume / weighted_ask_volume if weighted_ask_volume > 0 else 999
        else:
            best_bid = 0
            best_ask = 0
            weighted_imbalance = 1.0
        
        # Calculate spread
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0
        relative_spread = spread / best_bid if best_bid > 0 else 0
        
        # Result
        result = {
            "imbalance_ratio": imbalance_ratio,
            "weighted_imbalance": weighted_imbalance,
            "spread": spread,
            "relative_spread": relative_spread,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "weighted_bid_volume": weighted_bid_volume,
            "weighted_ask_volume": weighted_ask_volume,
            "timestamp": order_book["timestamp"]
        }
        
        # Add to history
        self.imbalance_history.append(result)
        
        return result
    
    def calculate_liquidity(self, order_book=None, price_range_pct=0.001):
        """
        Analyzes liquidity in the order book.
        
        Args:
            order_book: Order book data (optional, retrieves new data if None)
            price_range_pct: Price range to analyze as percentage of current price
            
        Returns:
            dict: Liquidity metrics
        """
        if order_book is None:
            order_book = self.get_order_book()
        
        if order_book is None:
            return None
        
        bids = order_book["bids"]
        asks = order_book["asks"]
        
        if not bids or not asks:
            return None
        
        # Current mid price
        mid_price = (bids[0]["price"] + asks[0]["price"]) / 2
        
        # Price ranges
        bid_range = mid_price * (1 - price_range_pct)
        ask_range = mid_price * (1 + price_range_pct)
        
        # Calculate liquidity within range
        bid_liquidity = sum(bid["volume"] for bid in bids if bid["price"] >= bid_range)
        ask_liquidity = sum(ask["volume"] for ask in asks if ask["price"] <= ask_range)
        
        # Calculate liquidity distribution (how evenly distributed the liquidity is)
        bid_volumes = [bid["volume"] for bid in bids]
        ask_volumes = [ask["volume"] for ask in asks]
        
        # Gini coefficient as measure of distribution (lower = more even)
        bid_gini = self._gini_coefficient(bid_volumes)
        ask_gini = self._gini_coefficient(ask_volumes)
        
        # Looking for "walls" - large orders compared to surrounding orders
        bid_walls = self._detect_walls(bids, threshold=3.0)
        ask_walls = self._detect_walls(asks, threshold=3.0)
        
        result = {
            "bid_liquidity": bid_liquidity,
            "ask_liquidity": ask_liquidity,
            "liquidity_ratio": bid_liquidity / ask_liquidity if ask_liquidity > 0 else 999,
            "bid_gini": bid_gini,
            "ask_gini": ask_gini,
            "liquidity_concentration": (bid_gini + ask_gini) / 2,
            "bid_walls": bid_walls,
            "ask_walls": ask_walls,
            "timestamp": order_book["timestamp"]
        }
        
        # Add to history
        self.liquidity_history.append(result)
        
        return result
    
    def get_institutional_activity(self):
        """
        Analyzes the order book for signs of institutional activity.
        
        Returns:
            dict: Metrics indicating institutional presence
        """
        order_book = self.get_order_book()
        
        if order_book is None:
            return None
        
        # Get imbalance and liquidity metrics
        imbalance = self.calculate_imbalance(order_book)
        liquidity = self.calculate_liquidity(order_book)
        
        if imbalance is None or liquidity is None:
            return None
        
        # Analyze imbalance shifts (comparing to historical values)
        imbalance_shift = 0
        if len(self.imbalance_history) > 5:
            prev_imbalances = [h["imbalance_ratio"] for h in list(self.imbalance_history)[-6:-1]]
            avg_prev_imbalance = sum(prev_imbalances) / len(prev_imbalances)
            imbalance_shift = imbalance["imbalance_ratio"] - avg_prev_imbalance
        
        # Check for iceberg orders (hidden liquidity)
        iceberg_probability = 0.0
        
        # If recent trades show larger volumes than visible in the book
        # This requires trade data which MT5 doesn't directly provide
        # We'll approximate using changes in order book between snapshots
        if len(self.history) > 1:
            prev_order_book = self.history[-2]
            current_order_book = self.history[-1]
            
            # Compare changes in liquidity
            if prev_order_book and "bids" in prev_order_book and "asks" in prev_order_book:
                prev_total_bid = sum(bid["volume"] for bid in prev_order_book["bids"])
                prev_total_ask = sum(ask["volume"] for ask in prev_order_book["asks"])
                
                current_total_bid = sum(bid["volume"] for bid in current_order_book["bids"])
                current_total_ask = sum(ask["volume"] for ask in current_order_book["asks"])
                
                # Large inconsistent changes might indicate hidden liquidity
                bid_change = abs(current_total_bid - prev_total_bid) / prev_total_bid if prev_total_bid > 0 else 0
                ask_change = abs(current_total_ask - prev_total_ask) / prev_total_ask if prev_total_ask > 0 else 0
                
                if bid_change > 0.2 or ask_change > 0.2:  # 20% change threshold
                    iceberg_probability = max(bid_change, ask_change)
        
        # Check for smart money divergence (when price moves against the imbalance)
        smart_money_divergence = 0.0
        if len(self.imbalance_history) > 2 and len(self.history) > 2:
            prev_imbalance = self.imbalance_history[-2]["imbalance_ratio"]
            current_imbalance = imbalance["imbalance_ratio"]
            
            # Get price change
            prev_mid = (self.history[-2]["bids"][0]["price"] + self.history[-2]["asks"][0]["price"]) / 2
            current_mid = (order_book["bids"][0]["price"] + order_book["asks"][0]["price"]) / 2
            price_change = current_mid - prev_mid
            
            # If imbalance favors bids but price is falling, or vice versa
            if (prev_imbalance > 1.2 and price_change < 0) or (prev_imbalance < 0.8 and price_change > 0):
                smart_money_divergence = abs(price_change / prev_mid)
        
        # Combine metrics into institutional activity score
        institutional_score = 0.0
        
        # High concentration of liquidity
        if liquidity["liquidity_concentration"] > 0.7:  # High Gini coefficient
            institutional_score += 0.3
        
        # Presence of walls
        if liquidity["bid_walls"] or liquidity["ask_walls"]:
            institutional_score += 0.2
        
        # Large imbalance shifts
        if abs(imbalance_shift) > 1.0:
            institutional_score += 0.2
        
        # Iceberg orders
        institutional_score += iceberg_probability * 0.3
        
        # Smart money divergence
        institutional_score += smart_money_divergence * 0.2
        
        # Cap at 1.0
        institutional_score = min(1.0, institutional_score)
        
        result = {
            "institutional_score": institutional_score,
            "imbalance_shift": imbalance_shift,
            "iceberg_probability": iceberg_probability,
            "smart_money_divergence": smart_money_divergence,
            "liquidity_walls": len(liquidity["bid_walls"]) + len(liquidity["ask_walls"]),
            "timestamp": order_book["timestamp"]
        }
        
        return result
    
    def get_optimal_execution_time(self, side="buy", volume=0.1):
        """
        Determines the optimal time to execute a trade based on order book conditions.
        
        Args:
            side: "buy" or "sell"
            volume: Trade volume in lots
            
        Returns:
            dict: Execution recommendation
        """
        # Check recent order book history for patterns
        if len(self.imbalance_history) < 10:
            return {"recommendation": "insufficient_data"}
        
        # Get current metrics
        order_book = self.get_order_book()
        imbalance = self.calculate_imbalance(order_book)
        liquidity = self.calculate_liquidity(order_book)
        
        if not imbalance or not liquidity:
            return {"recommendation": "insufficient_data"}
        
        # Favorable conditions for BUY
        buy_favorable = False
        buy_reason = []
        
        if side.lower() == "buy":
            # Favorable: Higher ask liquidity, lower bid/ask imbalance
            if imbalance["imbalance_ratio"] < 0.8:
                buy_favorable = True
                buy_reason.append("bid/ask imbalance favors buying")
            
            # Favorable: No sell walls nearby
            if not liquidity["ask_walls"]:
                buy_favorable = True
                buy_reason.append("no significant sell walls")
            
            # Favorable: Low spread
            if imbalance["relative_spread"] < 0.0003:  # 0.03% spread
                buy_favorable = True
                buy_reason.append("tight spread")
        
        # Favorable conditions for SELL
        sell_favorable = False
        sell_reason = []
        
        if side.lower() == "sell":
            # Favorable: Higher bid liquidity, higher bid/ask imbalance
            if imbalance["imbalance_ratio"] > 1.2:
                sell_favorable = True
                sell_reason.append("bid/ask imbalance favors selling")
            
            # Favorable: No buy walls nearby
            if not liquidity["bid_walls"]:
                sell_favorable = True
                sell_reason.append("no significant buy walls")
            
            # Favorable: Low spread
            if imbalance["relative_spread"] < 0.0003:  # 0.03% spread
                sell_favorable = True
                sell_reason.append("tight spread")
        
        # Make recommendation
        if side.lower() == "buy" and buy_favorable:
            return {
                "recommendation": "favorable",
                "reasons": buy_reason,
                "expected_slippage": "low",
                "timestamp": datetime.now()
            }
        elif side.lower() == "sell" and sell_favorable:
            return {
                "recommendation": "favorable",
                "reasons": sell_reason,
                "expected_slippage": "low",
                "timestamp": datetime.now()
            }
        else:
            return {
                "recommendation": "unfavorable",
                "side": side,
                "timestamp": datetime.now()
            }
    
    def get_features(self):
        """
        Extracts order book features for use in the HMM model.
        
        Returns:
            dict: Features derived from order book analysis
        """
        # Get fresh metrics
        order_book = self.get_order_book()
        
        if order_book is None:
            return None
        
        imbalance = self.calculate_imbalance(order_book)
        liquidity = self.calculate_liquidity(order_book)
        institutional = self.get_institutional_activity()
        
        # Feature extraction
        features = {}
        
        # Basic imbalance features
        if imbalance:
            features["ob_imbalance"] = imbalance["imbalance_ratio"]
            features["ob_weighted_imbalance"] = imbalance["weighted_imbalance"]
            features["ob_relative_spread"] = imbalance["relative_spread"]
        
        # Liquidity features
        if liquidity:
            features["ob_liquidity_ratio"] = liquidity["liquidity_ratio"]
            features["ob_concentration"] = liquidity["liquidity_concentration"]
            features["ob_walls"] = 1 if (liquidity["bid_walls"] or liquidity["ask_walls"]) else 0
        
        # Institutional activity
        if institutional:
            features["ob_institutional"] = institutional["institutional_score"]
            features["ob_imbalance_shift"] = institutional["imbalance_shift"]
            features["ob_smart_money"] = institutional["smart_money_divergence"]
        
        # Feature normalization (simple z-score if history available)
        if len(self.imbalance_history) > 30:
            # Imbalance normalization
            imbalance_values = [h["imbalance_ratio"] for h in self.imbalance_history if "imbalance_ratio" in h]
            if imbalance_values:
                mean_imbalance = sum(imbalance_values) / len(imbalance_values)
                std_imbalance = np.std(imbalance_values) if len(imbalance_values) > 1 else 1.0
                if "ob_imbalance" in features:
                    features["ob_imbalance_zscore"] = (features["ob_imbalance"] - mean_imbalance) / std_imbalance
        
        # Signal-specific features
        features["ob_buy_signal"] = 0
        features["ob_sell_signal"] = 0
        
        # Check for strong buy/sell signals
        if imbalance and liquidity:
            if imbalance["imbalance_ratio"] < 0.7 and not liquidity["ask_walls"]:
                features["ob_buy_signal"] = 1
            elif imbalance["imbalance_ratio"] > 1.3 and not liquidity["bid_walls"]:
                features["ob_sell_signal"] = 1
        
        # --- FIX: Ensure fixed feature set and order --- 
        # Define the exact 20 features expected by the anomaly detector (adjust names/order as needed)
        expected_feature_keys = [
            'ob_imbalance', 'ob_weighted_imbalance', 'ob_relative_spread', 
            'ob_liquidity_ratio', 'ob_concentration', 'ob_walls',
            'ob_institutional', 'ob_imbalance_shift', 'ob_smart_money',
            'ob_imbalance_zscore',
            'ob_bid_liquidity_in_range', 'ob_ask_liquidity_in_range', # Renamed/added from liquidity dict
            'ob_bid_gini', 'ob_ask_gini', # Renamed/added from liquidity dict
            'ob_bid_wall_count', 'ob_ask_wall_count', # Added features
            'ob_buy_signal', 'ob_sell_signal', 
            'ob_reserved_1', 'ob_reserved_2' # Padding/Reserved features to reach 20
        ]
        
        final_features = {}
        # Populate with calculated features or defaults
        final_features['ob_imbalance'] = imbalance.get("imbalance_ratio", 1.0) if imbalance else 1.0
        final_features['ob_weighted_imbalance'] = imbalance.get("weighted_imbalance", 1.0) if imbalance else 1.0
        final_features['ob_relative_spread'] = imbalance.get("relative_spread", 0.0) if imbalance else 0.0
        final_features['ob_liquidity_ratio'] = liquidity.get("liquidity_ratio", 1.0) if liquidity else 1.0
        final_features['ob_concentration'] = liquidity.get("liquidity_concentration", 0.5) if liquidity else 0.5
        final_features['ob_walls'] = 1 if (liquidity and (liquidity.get("bid_walls") or liquidity.get("ask_walls"))) else 0
        final_features['ob_institutional'] = institutional.get("institutional_score", 0.0) if institutional else 0.0
        final_features['ob_imbalance_shift'] = institutional.get("imbalance_shift", 0.0) if institutional else 0.0
        final_features['ob_smart_money'] = institutional.get("smart_money_divergence", 0.0) if institutional else 0.0
        final_features['ob_imbalance_zscore'] = features.get("ob_imbalance_zscore", 0.0) # Calculated above if possible
        final_features['ob_bid_liquidity_in_range'] = liquidity.get("bid_liquidity", 0.0) if liquidity else 0.0
        final_features['ob_ask_liquidity_in_range'] = liquidity.get("ask_liquidity", 0.0) if liquidity else 0.0
        final_features['ob_bid_gini'] = liquidity.get("bid_gini", 0.5) if liquidity else 0.5
        final_features['ob_ask_gini'] = liquidity.get("ask_gini", 0.5) if liquidity else 0.5
        final_features['ob_bid_wall_count'] = len(liquidity.get("bid_walls", [])) if liquidity else 0
        final_features['ob_ask_wall_count'] = len(liquidity.get("ask_walls", [])) if liquidity else 0
        final_features['ob_buy_signal'] = features.get("ob_buy_signal", 0)
        final_features['ob_sell_signal'] = features.get("ob_sell_signal", 0)
        final_features['ob_reserved_1'] = 0.0 # Padding
        final_features['ob_reserved_2'] = 0.0 # Padding
        
        # Verify the final count
        if len(final_features) != 20:
            self.logger.error(f"Feature count mismatch in get_features! Expected 20, got {len(final_features)}")
            # Fallback to return default 20 zeros if error
            return {key: 0.0 for key in expected_feature_keys}
        
        return final_features
        # --- END FIX ---
    
    def _gini_coefficient(self, values):
        """
        Calculates Gini coefficient as a measure of distribution inequality.
        """
        if not values or len(values) < 2 or sum(values) == 0:
            return 0
        
        values = sorted(values)
        n = len(values)
        cumsum = 0
        for i, value in enumerate(values):
            cumsum += (i + 1) * value
        
        # Formula for Gini coefficient
        return (2 * cumsum) / (n * sum(values)) - (n + 1) / n
    
    def _detect_walls(self, orders, threshold=3.0):
        """
        Detects "walls" in the order book (large orders compared to surrounding orders).
        
        Args:
            orders: List of order dictionaries with price and volume
            threshold: Multiple of average volume to consider as a wall
            
        Returns:
            list: List of detected walls with price and volume
        """
        if not orders or len(orders) < 3:
            return []
        
        volumes = [order["volume"] for order in orders]
        avg_volume = sum(volumes) / len(volumes)
        
        walls = []
        
        for i, order in enumerate(orders):
            # Compare to local average (surrounding 3 orders if possible)
            start_idx = max(0, i-1)
            end_idx = min(len(orders), i+2)
            
            if end_idx - start_idx < 2:  # Need at least 2 orders for comparison
                continue
            
            local_volumes = [orders[j]["volume"] for j in range(start_idx, end_idx) if j != i]
            local_avg = sum(local_volumes) / len(local_volumes)
            
            # Check if this order is significantly larger than local average
            if order["volume"] > threshold * local_avg and order["volume"] > threshold * avg_volume:
                walls.append({
                    "price": order["price"],
                    "volume": order["volume"],
                    "relative_size": order["volume"] / local_avg
                })
        
        return walls