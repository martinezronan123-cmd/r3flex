#!/usr/bin/env python3
"""
R3flex Traffic Analyzer - Advanced Network Intelligence & Pattern Detection
Real-Time Deep Packet Inspection with Machine Learning Anomaly Detection
Version: 3.1.0 | Analysis Depth: Military-Grade Traffic Intelligence
"""

import asyncio
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
import zlib
import re
import ipaddress
import statistics
import math
from concurrent.futures import ThreadPoolExecutor
import time
import psutil
import socket
import struct

class TrafficType(Enum):
    """Advanced traffic classification"""
    HTTP_GET = "http_get"
    HTTP_POST = "http_post"
    HTTPS_TUNNEL = "https_tunnel"
    DNS_QUERY = "dns_query"
    API_CALL = "api_call"
    STREAMING = "streaming"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    PROXY_TUNNEL = "proxy_tunnel"
    ENCRYPTED = "encrypted"
    COMPRESSED = "compressed"

class AnomalyType(Enum):
    """Traffic anomaly classification"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    BLOCKED = "blocked"
    THROTTLED = "throttled"
    FILTERED = "filtered"
    INJECTED = "injected"
    MIM_ATTACK = "mitm_attack"

@dataclass
class TrafficSample:
    """Comprehensive traffic sample with deep metadata"""
    sample_id: str
    timestamp: datetime
    source_ip: str
    destination_ip: str
    protocol: str
    packet_size: int
    payload_hash: str
    traffic_type: TrafficType
    response_time: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    anomaly_score: float = 0.0
    anomaly_type: AnomalyType = AnomalyType.NORMAL

@dataclass
class TrafficPattern:
    """Identified traffic pattern with behavioral characteristics"""
    pattern_id: str
    signature: str
    frequency: int
    success_rate: float
    average_response_time: float
    typical_payload_size: int
    associated_methods: List[str]
    risk_level: int
    last_observed: datetime

@dataclass
class NetworkIntelligence:
    """Comprehensive network intelligence report"""
    timestamp: datetime
    traffic_volume: int
    anomaly_count: int
    threat_level: int
    detected_filters: List[str]
    performance_metrics: Dict[str, float]
    security_recommendations: List[str]
    pattern_analysis: Dict[str, Any]

class DeepPacketInspector:
    """Advanced Deep Packet Inspection with ML-based classification"""
    
    def __init__(self):
        self.pattern_database = {}
        self.signature_regex = self._load_signature_regex()
        self.ml_model = self._initialize_ml_model()
        self.scaler = StandardScaler()
        
    def _load_signature_regex(self) -> Dict[str, re.Pattern]:
        """Load advanced signature patterns for traffic classification"""
        return {
            'blocked_pattern': re.compile(r'(blocked|denied|forbidden|access.denied)', re.IGNORECASE),
            'captcha_pattern': re.compile(r'(captcha|recaptcha|verification)', re.IGNORECASE),
            'filter_pattern': re.compile(r'(filter|firewall|proxy.detected)', re.IGNORECASE),
            'throttle_pattern': re.compile(r'(rate.limit|throttle|slow.down)', re.IGNORECASE),
            'injection_pattern': re.compile(r'(script|alert|document\.cookie)', re.IGNORECASE),
            'success_pattern': re.compile(r'(200 OK|success|ok)', re.IGNORECASE),
            'redirect_pattern': re.compile(r'(30[12]|redirect|location:)', re.IGNORECASE)
        }
    
    def _initialize_ml_model(self) -> Any:
        """Initialize machine learning model for traffic classification"""
        try:
            # Use a pre-trained model or create a new one
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(50,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(7, activation='softmax')  # 7 anomaly types
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        except Exception as e:
            logging.warning(f"ML model initialization failed: {e}")
            return None
    
    async def analyze_packet(self, packet_data: bytes, metadata: Dict[str, Any]) -> TrafficSample:
        """Perform deep packet inspection and analysis"""
        sample_id = hashlib.md5(packet_data).hexdigest()[:16]
        
        # Extract basic packet information
        packet_size = len(packet_data)
        payload_hash = hashlib.sha256(packet_data).hexdigest()
        
        # Advanced packet parsing
        protocol = self._detect_protocol(packet_data)
        traffic_type = self._classify_traffic_type(packet_data, metadata)
        
        # Create traffic sample
        sample = TrafficSample(
            sample_id=sample_id,
            timestamp=datetime.now(),
            source_ip=metadata.get('source_ip', '0.0.0.0'),
            destination_ip=metadata.get('dest_ip', '0.0.0.0'),
            protocol=protocol,
            packet_size=packet_size,
            payload_hash=payload_hash,
            traffic_type=traffic_type,
            response_time=metadata.get('response_time', 0.0),
            success=metadata.get('success', False)
        )
        
        # Perform deep analysis
        await self._perform_deep_analysis(sample, packet_data, metadata)
        
        return sample
    
    def _detect_protocol(self, packet_data: bytes) -> str:
        """Detect network protocol from packet data"""
        if len(packet_data) < 20:
            return "unknown"
        
        try:
            # Ethernet frame detection
            if packet_data[12:14] == b'\x08\x00':  # IPv4
                ip_header = packet_data[14:34]
                protocol = ip_header[9]  # Protocol field
                
                if protocol == 6:
                    return "TCP"
                elif protocol == 17:
                    return "UDP"
                elif protocol == 1:
                    return "ICMP"
            
            # HTTP detection
            if packet_data.startswith(b'GET') or packet_data.startswith(b'POST'):
                return "HTTP"
            elif packet_data.startswith(b'HTTP'):
                return "HTTP_RESPONSE"
                
        except Exception:
            pass
        
        return "encrypted"
    
    def _classify_traffic_type(self, packet_data: bytes, metadata: Dict[str, Any]) -> TrafficType:
        """Classify traffic type using advanced pattern recognition"""
        content = packet_data.decode('utf-8', errors='ignore').lower()
        
        # Check for specific patterns
        for pattern_name, pattern in self.signature_regex.items():
            if pattern.search(content):
                if pattern_name == 'blocked_pattern':
                    return TrafficType.ENCRYPTED  # Likely encrypted response
                elif pattern_name == 'success_pattern':
                    return TrafficType.HTTP_GET  # Successful request
        
        # Size-based classification
        packet_size = len(packet_data)
        if packet_size < 100:
            return TrafficType.DNS_QUERY
        elif packet_size > 100000:
            return TrafficType.DOWNLOAD
        elif 'api' in metadata.get('url', '').lower():
            return TrafficType.API_CALL
        
        return TrafficType.HTTP_GET
    
    async def _perform_deep_analysis(self, sample: TrafficSample, packet_data: bytes, metadata: Dict[str, Any]):
        """Perform deep analysis including anomaly detection"""
        # Feature extraction for ML analysis
        features = self._extract_features(sample, packet_data, metadata)
        
        # Anomaly detection
        anomaly_score = await self._detect_anomaly(features)
        sample.anomaly_score = anomaly_score
        
        # Classify anomaly type
        sample.anomaly_type = self._classify_anomaly(anomaly_score, features, metadata)
        
        # Pattern recognition
        await self._update_pattern_database(sample, features)
    
    def _extract_features(self, sample: TrafficSample, packet_data: bytes, metadata: Dict[str, Any]) -> List[float]:
        """Extract advanced features for ML analysis"""
        features = []
        
        # Basic features
        features.extend([
            sample.packet_size,
            sample.response_time,
            1.0 if sample.success else 0.0,
            len(sample.source_ip.split('.')),
            len(sample.destination_ip.split('.'))
        ])
        
        # Protocol features
        protocol_map = {'TCP': 1, 'UDP': 2, 'HTTP': 3, 'encrypted': 4}
        features.append(protocol_map.get(sample.protocol, 0))
        
        # Content-based features
        content = packet_data.decode('utf-8', errors='ignore')
        features.extend([
            len(content),
            content.count(' '),  # Word count estimate
            content.count('\n'),  # Line count
            len(re.findall(r'[0-9]', content)),  # Digit count
            len(re.findall(r'[a-zA-Z]', content))  # Letter count
        ])
        
        # Advanced statistical features
        if len(packet_data) > 0:
            byte_values = [b for b in packet_data]
            features.extend([
                statistics.mean(byte_values),
                statistics.stdev(byte_values) if len(byte_values) > 1 else 0,
                max(byte_values),
                min(byte_values)
            ])
        
        # Fill to 50 features with derived values
        while len(features) < 50:
            features.append(features[len(features) % len(features)] * 0.5)
        
        return features[:50]
    
    async def _detect_anomaly(self, features: List[float]) -> float:
        """Detect anomalies using ensemble ML methods"""
        try:
            features_array = np.array(features).reshape(1, -1)
            
            # Use multiple detection methods
            scores = []
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_score = iso_forest.fit_predict(features_array)[0]
            scores.append(1.0 if iso_score == -1 else 0.0)
            
            # One-Class SVM
            oc_svm = OneClassSVM(gamma='auto', nu=0.1)
            svm_score = oc_svm.fit_predict(features_array)[0]
            scores.append(1.0 if svm_score == -1 else 0.0)
            
            # Statistical outlier detection
            z_scores = np.abs(stats.zscore(features_array))
            stat_score = np.mean(z_scores > 3)  # Points beyond 3 std dev
            scores.append(stat_score)
            
            # Return average anomaly score
            return np.mean(scores)
            
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")
            return 0.0
    
    def _classify_anomaly(self, anomaly_score: float, features: List[float], metadata: Dict[str, Any]) -> AnomalyType:
        """Classify the type of anomaly detected"""
        if anomaly_score < 0.3:
            return AnomalyType.NORMAL
        elif anomaly_score < 0.6:
            return AnomalyType.SUSPICIOUS
        elif anomaly_score < 0.8:
            # Check for specific blockage patterns
            if metadata.get('response_time', 0) > 10.0:
                return AnomalyType.THROTTLED
            elif not metadata.get('success', True):
                return AnomalyType.BLOCKED
            else:
                return AnomalyType.MALICIOUS
        else:
            return AnomalyType.MALICIOUS

class BehavioralAnalyst:
    """Advanced behavioral analysis for traffic patterns"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.traffic_history: Deque[TrafficSample] = deque(maxlen=window_size)
        self.patterns: Dict[str, TrafficPattern] = {}
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        
    async def analyze_behavior(self, sample: TrafficSample) -> Dict[str, Any]:
        """Analyze behavioral patterns in traffic"""
        self.traffic_history.append(sample)
        
        analysis = {
            'behavioral_score': 0.0,
            'pattern_matches': [],
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Temporal analysis
        temporal_score = await self._temporal_analysis(sample)
        analysis['behavioral_score'] += temporal_score * 0.3
        
        # Volume analysis
        volume_score = await self._volume_analysis(sample)
        analysis['behavioral_score'] += volume_score * 0.3
        
        # Pattern matching
        pattern_matches = await self._pattern_matching(sample)
        analysis['pattern_matches'] = pattern_matches
        analysis['behavioral_score'] += len(pattern_matches) * 0.2
        
        # Clustering analysis
        cluster_score = await self._clustering_analysis(sample)
        analysis['behavioral_score'] += cluster_score * 0.2
        
        # Risk assessment
        analysis['risk_assessment'] = await self._risk_assessment(sample, analysis['behavioral_score'])
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    async def _temporal_analysis(self, sample: TrafficSample) -> float:
        """Analyze temporal patterns and timing anomalies"""
        if len(self.traffic_history) < 10:
            return 0.5  # Neutral score
        
        # Calculate request frequency
        recent_samples = list(self.traffic_history)[-10:]
        time_diffs = []
        
        for i in range(1, len(recent_samples)):
            diff = (recent_samples[i].timestamp - recent_samples[i-1].timestamp).total_seconds()
            time_diffs.append(diff)
        
        if not time_diffs:
            return 0.5
        
        avg_frequency = statistics.mean(time_diffs)
        current_frequency = (sample.timestamp - recent_samples[-1].timestamp).total_seconds()
        
        # Score based on frequency deviation
        if avg_frequency == 0:
            return 0.0
        
        deviation = abs(current_frequency - avg_frequency) / avg_frequency
        return max(0.0, 1.0 - deviation)  # Lower deviation = higher score
    
    async def _volume_analysis(self, sample: TrafficSample) -> float:
        """Analyze traffic volume patterns"""
        if len(self.traffic_history) < 20:
            return 0.5
        
        # Analyze volume trends
        recent_volumes = [s.packet_size for s in list(self.traffic_history)[-20:]]
        avg_volume = statistics.mean(recent_volumes)
        std_volume = statistics.stdev(recent_volumes) if len(recent_volumes) > 1 else 0
        
        if std_volume == 0:
            return 0.5
        
        # Z-score based analysis
        z_score = abs(sample.packet_size - avg_volume) / std_volume
        
        if z_score < 2:
            return 0.8  # Normal volume
        elif z_score < 3:
            return 0.5  # Slightly abnormal
        else:
            return 0.2  # Highly abnormal
    
    async def _pattern_matching(self, sample: TrafficSample) -> List[str]:
        """Match against known traffic patterns"""
        matches = []
        
        for pattern_id, pattern in self.patterns.items():
            similarity = self._calculate_pattern_similarity(sample, pattern)
            if similarity > 0.7:  # 70% similarity threshold
                matches.append(pattern_id)
        
        return matches
    
    def _calculate_pattern_similarity(self, sample: TrafficSample, pattern: TrafficPattern) -> float:
        """Calculate similarity between sample and known pattern"""
        similarity = 0.0
        
        # Size similarity
        size_diff = abs(sample.packet_size - pattern.typical_payload_size)
        size_similarity = max(0.0, 1.0 - (size_diff / max(sample.packet_size, pattern.typical_payload_size)))
        similarity += size_similarity * 0.3
        
        # Response time similarity
        time_diff = abs(sample.response_time - pattern.average_response_time)
        if pattern.average_response_time > 0:
            time_similarity = max(0.0, 1.0 - (time_diff / pattern.average_response_time))
            similarity += time_similarity * 0.3
        
        # Success rate consideration
        if sample.success and pattern.success_rate > 0.8:
            similarity += 0.2
        elif not sample.success and pattern.success_rate < 0.2:
            similarity += 0.2
        
        # Traffic type matching
        if sample.traffic_type.name.lower() in pattern.signature.lower():
            similarity += 0.2
        
        return min(1.0, similarity)
    
    async def _clustering_analysis(self, sample: TrafficSample) -> float:
        """Perform clustering analysis to detect novel patterns"""
        if len(self.traffic_history) < 50:
            return 0.5
        
        # Extract features for clustering
        features = []
        for hist_sample in list(self.traffic_history)[-50:]:
            feature_vector = [
                hist_sample.packet_size,
                hist_sample.response_time,
                1.0 if hist_sample.success else 0.0
            ]
            features.append(feature_vector)
        
        features.append([sample.packet_size, sample.response_time, 1.0 if sample.success else 0.0])
        
        try:
            # Perform clustering
            clusters = self.clustering_model.fit_predict(features)
            sample_cluster = clusters[-1]
            
            # Count samples in the same cluster
            cluster_size = np.sum(clusters == sample_cluster)
            cluster_score = cluster_size / len(features)
            
            return cluster_score  # Higher = more common pattern
            
        except Exception as e:
            logging.warning(f"Clustering analysis failed: {e}")
            return 0.5
    
    async def _risk_assessment(self, sample: TrafficSample, behavioral_score: float) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        risk_factors = []
        overall_risk = 0.0
        
        # Anomaly-based risk
        if sample.anomaly_score > 0.7:
            risk_factors.append(f"High anomaly score: {sample.anomaly_score:.2f}")
            overall_risk += sample.anomaly_score * 0.4
        
        # Behavioral risk
        behavioral_risk = 1.0 - behavioral_score
        if behavioral_risk > 0.5:
            risk_factors.append(f"Abnormal behavior: {behavioral_risk:.2f}")
            overall_risk += behavioral_risk * 0.3
        
        # Content-based risk
        content_risk = await self._content_risk_analysis(sample)
        if content_risk > 0.6:
            risk_factors.append(f"Suspicious content: {content_risk:.2f}")
            overall_risk += content_risk * 0.3
        
        return {
            'overall_risk': min(1.0, overall_risk),
            'risk_factors': risk_factors,
            'risk_level': self._classify_risk_level(overall_risk)
        }
    
    async def _content_risk_analysis(self, sample: TrafficSample) -> float:
        """Analyze content for potential risks"""
        # This would analyze the actual content for malicious patterns
        # For now, return a basic risk assessment based on anomaly type
        risk_mapping = {
            AnomalyType.NORMAL: 0.1,
            AnomalyType.SUSPICIOUS: 0.4,
            AnomalyType.MALICIOUS: 0.9,
            AnomalyType.BLOCKED: 0.7,
            AnomalyType.THROTTLED: 0.6,
            AnomalyType.FILTERED: 0.8,
            AnomalyType.INJECTED: 1.0,
            AnomalyType.MIM_ATTACK: 1.0
        }
        
        return risk_mapping.get(sample.anomaly_type, 0.5)
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []
        risk_level = analysis['risk_assessment']['risk_level']
        
        if risk_level in ["HIGH", "CRITICAL"]:
            recommendations.extend([
                "Consider switching proxy method immediately",
                "Enable additional encryption layers",
                "Reduce request frequency temporarily",
                "Verify destination server authenticity"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Monitor traffic patterns closely",
                "Consider method rotation",
                "Enable additional logging"
            ])
        
        if analysis['behavioral_score'] < 0.4:
            recommendations.append("Behavioral anomaly detected - review patterns")
        
        return recommendations

class PerformanceMonitor:
    """Advanced performance monitoring and optimization"""
    
    def __init__(self):
        self.performance_metrics = defaultdict(list)
        self.baseline_metrics = {}
        self.optimization_recommendations = []
        
    async def monitor_performance(self, sample: TrafficSample) -> Dict[str, Any]:
        """Monitor and analyze performance metrics"""
        metrics = {
            'throughput': await self._calculate_throughput(),
            'latency': sample.response_time,
            'success_rate': await self._calculate_success_rate(),
            'efficiency': await self._calculate_efficiency(sample),
            'resource_usage': await self._get_resource_usage()
        }
        
        # Store metrics for trend analysis
        for key, value in metrics.items():
            self.performance_metrics[key].append(value)
            # Keep only last 100 samples
            if len(self.performance_metrics[key]) > 100:
                self.performance_metrics[key] = self.performance_metrics[key][-100:]
        
        # Performance analysis
        analysis = {
            'current_metrics': metrics,
            'trend_analysis': await self._analyze_trends(),
            'bottlenecks': await self._identify_bottlenecks(metrics),
            'optimization_opportunities': await self._find_optimizations(metrics)
        }
        
        return analysis
    
    async def _calculate_throughput(self) -> float:
        """Calculate current throughput in requests per second"""
        if len(self.performance_metrics['latency']) < 2:
            return 0.0
        
        recent_latencies = self.performance_metrics['latency'][-10:]
        avg_latency = statistics.mean(recent_latencies) if recent_latencies else 1.0
        
        if avg_latency > 0:
            return 1.0 / avg_latency
        return 0.0
    
    async def _calculate_success_rate(self) -> float:
        """Calculate recent success rate"""
        if not self.performance_metrics.get('success'):
            return 1.0
        
        recent_successes = self.performance_metrics['success'][-20:]
        if not recent_successes:
            return 1.0
        
        success_count = sum(1 for s in recent_successes if s)
        return success_count / len(recent_successes)
    
    async def _calculate_efficiency(self, sample: TrafficSample) -> float:
        """Calculate traffic efficiency score"""
        # Efficiency based on payload size vs overhead
        if sample.packet_size == 0:
            return 0.0
        
        # Ideal efficiency ratio (adjust based on protocol)
        ideal_ratio = 0.8  # 80% payload, 20% overhead
        
        # Calculate actual efficiency (simplified)
        headers_size = len(str(sample.metadata).encode()) if sample.metadata else 100
        payload_efficiency = sample.packet_size / (sample.packet_size + headers_size)
        
        return min(1.0, payload_efficiency / ideal_ratio)
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get system resource usage metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io': psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes,
            'network_io': psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        }
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.performance_metrics['latency']) < 10:
            return {'trend': 'insufficient_data', 'confidence': 0.0}
        
        # Simple linear trend analysis
        latencies = self.performance_metrics['latency'][-20:]
        x = list(range(len(latencies)))
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, latencies)
            
            trend = "improving" if slope < -0.1 else "degrading" if slope > 0.1 else "stable"
            confidence = abs(r_value)
            
            return {
                'trend': trend,
                'confidence': confidence,
                'slope': slope,
                'r_squared': r_value**2
            }
        except:
            return {'trend': 'unknown', 'confidence': 0.0}
    
    async def _identify_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Latency bottleneck
        if metrics['latency'] > 5.0:  # More than 5 seconds
            bottlenecks.append("High latency detected")
        
        # Success rate bottleneck
        if metrics['success_rate'] < 0.8:  # Less than 80% success
            bottlenecks.append("Low success rate affecting performance")
        
        # Resource bottlenecks
        if metrics['resource_usage']['cpu_percent'] > 80:
            bottlenecks.append("High CPU usage limiting performance")
        if metrics['resource_usage']['memory_percent'] > 85:
            bottlenecks.append("High memory usage affecting performance")
        
        return bottlenecks
    
    async def _find_optimizations(self, metrics: Dict[str, Any]) -> List[str]:
        """Find optimization opportunities"""
        optimizations = []
        
        if metrics['efficiency'] < 0.6:
            optimizations.append("Consider compression for better efficiency")
        
        if metrics['latency'] > 2.0:
            optimizations.append("Explore faster proxy methods or CDN")
        
        if metrics['success_rate'] < 0.9:
            optimizations.append("Implement better error handling and retry logic")
        
        return optimizations

class TrafficAnalyzer:
    """
    Advanced Traffic Analysis System with Real-Time Intelligence
    Enterprise-grade network monitoring and pattern detection
    """
    
    def __init__(self, sample_rate: float = 0.1, analysis_depth: str = "deep"):
        self.sample_rate = sample_rate
        self.analysis_depth = analysis_depth
        self.logger = logging.getLogger("TrafficAnalyzer")
        
        # Core analysis components
        self.packet_inspector = DeepPacketInspector()
        self.behavioral_analyst = BehavioralAnalyst()
        self.performance_monitor = PerformanceMonitor()
        
        # Data storage
        self.traffic_samples: Deque[TrafficSample] = deque(maxlen=10000)
        self.intelligence_reports: Deque[NetworkIntelligence] = deque(maxlen=100)
        
        # Analysis threads
        self.analysis_executor = ThreadPoolExecutor(max_workers=10)
        self.is_monitoring = False
        
        self.logger.info("Advanced Traffic Analyzer initialized")
    
    async def start_monitoring(self):
        """Start continuous traffic monitoring"""
        self.is_monitoring = True
        self.logger.info("Starting traffic monitoring...")
        
        # Start background monitoring tasks
        asyncio.create_task(self._continuous_analysis())
        asyncio.create_task(self._periodic_intelligence_reports())
        
        self.logger.info("Traffic monitoring active")
    
    async def stop_monitoring(self):
        """Stop traffic monitoring"""
        self.is_monitoring = False
        self.analysis_executor.shutdown(wait=True)
        self.logger.info("Traffic monitoring stopped")
    
    async def analyze_traffic(self, packet_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive traffic analysis entry point"""
        if random.random() > self.sample_rate:  # Apply sampling rate
            return {'sampled_out': True}
        
        try:
            # Deep packet inspection
            sample = await self.packet_inspector.analyze_packet(packet_data, metadata)
            self.traffic_samples.append(sample)
            
            # Behavioral analysis
            behavioral_analysis = await self.behavioral_analyst.analyze_behavior(sample)
            
            # Performance monitoring
            performance_analysis = await self.performance_monitor.monitor_performance(sample)
            
            # Compile comprehensive analysis
            analysis_report = {
                'sample_id': sample.sample_id,
                'timestamp': sample.timestamp.isoformat(),
                'basic_analysis': {
                    'traffic_type': sample.traffic_type.value,
                    'anomaly_score': sample.anomaly_score,
                    'anomaly_type': sample.anomaly_type.value,
                    'packet_size': sample.packet_size,
                    'response_time': sample.response_time
                },
                'behavioral_analysis': behavioral_analysis,
                'performance_analysis': performance_analysis,
                'security_assessment': {
                    'threat_level': behavioral_analysis['risk_assessment']['risk_level'],
                    'recommendations': behavioral_analysis['recommendations']
                }
            }
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Traffic analysis failed: {str(e)}")
            return {'error': str(e)}
    
    async def _continuous_analysis(self):
        """Continuous background analysis of traffic patterns"""
        while self.is_monitoring:
            try:
                # Perform batch analysis on recent samples
                if len(self.traffic_samples) > 100:
                    recent_samples = list(self.traffic_samples)[-100:]
                    await self._batch_pattern_analysis(recent_samples)
                
                await asyncio.sleep(10)  # Analyze every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Continuous analysis error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _batch_pattern_analysis(self, samples: List[TrafficSample]):
        """Perform batch analysis to identify patterns"""
        try:
            # Group samples by traffic type
            grouped_samples = defaultdict(list)
            for sample in samples:
                grouped_samples[sample.traffic_type].append(sample)
            
            # Analyze each group for patterns
            for traffic_type, type_samples in grouped_samples.items():
                if len(type_samples) >= 5:  # Minimum for pattern detection
                    await self._detect_patterns(traffic_type, type_samples)
                    
        except Exception as e:
            self.logger.error(f"Batch pattern analysis failed: {str(e)}")
    
    async def _detect_patterns(self, traffic_type: TrafficType, samples: List[TrafficSample]):
        """Detect and record traffic patterns"""
        if not samples:
            return
        
        # Calculate pattern characteristics
        success_rate = sum(1 for s in samples if s.success) / len(samples)
        avg_response_time = statistics.mean([s.response_time for s in samples if s.response_time > 0])
        avg_packet_size = statistics.mean([s.packet_size for s in samples])
        
        # Create pattern signature
        signature = f"{traffic_type.value}_success_{success_rate:.2f}_time_{avg_response_time:.2f}"
        pattern_id = hashlib.md5(signature.encode()).hexdigest()[:8]
        
        # Create or update pattern
        if pattern_id not in self.behavioral_analyst.patterns:
            pattern = TrafficPattern(
                pattern_id=pattern_id,
                signature=signature,
                frequency=len(samples),
                success_rate=success_rate,
                average_response_time=avg_response_time,
                typical_payload_size=int(avg_packet_size),
                associated_methods=list(set(s.metadata.get('method', '') for s in samples)),
                risk_level=self._calculate_pattern_risk(samples),
                last_observed=datetime.now()
            )
            self.behavioral_analyst.patterns[pattern_id] = pattern
        else:
            # Update existing pattern
            pattern = self.behavioral_analyst.patterns[pattern_id]
            pattern.frequency += len(samples)
            pattern.last_observed = datetime.now()
    
    def _calculate_pattern_risk(self, samples: List[TrafficSample]) -> int:
        """Calculate risk level for a pattern"""
        anomaly_scores = [s.anomaly_score for s in samples]
        avg_anomaly = statistics.mean(anomaly_scores) if anomaly_scores else 0
        
        if avg_anomaly < 0.3:
            return 1  # Low risk
        elif avg_anomaly < 0.6:
            return 2  # Medium risk
        elif avg_anomaly < 0.8:
            return 3  # High risk
        else:
            return 4  # Critical risk
    
    async def _periodic_intelligence_reports(self):
        """Generate periodic network intelligence reports"""
        report_interval = 300  # 5 minutes
        
        while self.is_monitoring:
            try:
                await asyncio.sleep(report_interval)
                
                if len(self.traffic_samples) > 0:
                    intelligence = await self._generate_intelligence_report()
                    self.intelligence_reports.append(intelligence)
                    
                    # Log high-level insights
                    if intelligence.threat_level > 2:
                        self.logger.warning(
                            f"Network intelligence alert: Threat level {intelligence.threat_level}"
                        )
                        
            except Exception as e:
                self.logger.error(f"Intelligence report generation failed: {str(e)}")
                await asyncio.sleep(60)
    
    async def _generate_intelligence_report(self) -> NetworkIntelligence:
        """Generate comprehensive network intelligence report"""
        recent_samples = list(self.traffic_samples)[-1000:]  # Last 1000 samples
        
        # Calculate basic metrics
        traffic_volume = sum(s.packet_size for s in recent_samples)
        anomaly_count = sum(1 for s in recent_samples if s.anomaly_score > 0.7)
        
        # Threat level calculation
        threat_level = self._calculate_overall_threat_level(recent_samples)
        
        # Detect filters and blocks
        detected_filters = await self._detect_filters(recent_samples)
        
        # Performance metrics
        performance_metrics = await self.performance_monitor.monitor_performance(recent_samples[-1] if recent_samples else None)
        
        # Pattern analysis
        pattern_analysis = {
            'total_patterns': len(self.behavioral_analyst.patterns),
            'recent_pattern_matches': await self._analyze_recent_patterns(recent_samples),
            'emerging_threats': await self._identify_emerging_threats(recent_samples)
        }
        
        # Security recommendations
        security_recommendations = self._generate_security_recommendations(
            threat_level, detected_filters, pattern_analysis
        )
        
        return NetworkIntelligence(
            timestamp=datetime.now(),
            traffic_volume=traffic_volume,
            anomaly_count=anomaly_count,
            threat_level=threat_level,
            detected_filters=detected_filters,
            performance_metrics=performance_metrics.get('current_metrics', {}),
            security_recommendations=security_recommendations,
            pattern_analysis=pattern_analysis
        )
    
    def _calculate_overall_threat_level(self, samples: List[TrafficSample]) -> int:
        """Calculate overall network threat level"""
        if not samples:
            return 1
        
        # Average anomaly score
        avg_anomaly = statistics.mean([s.anomaly_score for s in samples])
        
        # Percentage of high-risk samples
        high_risk_count = sum(1 for s in samples if s.anomaly_score > 0.7)
        high_risk_ratio = high_risk_count / len(samples)
        
        # Calculate threat level (1-5 scale)
        base_threat = min(5, int(avg_anomaly * 3 + high_risk_ratio * 2))
        
        return max(1, base_threat)
    
    async def _detect_filters(self, samples: List[TrafficSample]) -> List[str]:
        """Detect network filters and blocking mechanisms"""
        filters_detected = []
        
        # Analyze failure patterns
        failed_samples = [s for s in samples if not s.success]
        if failed_samples:
            failure_rate = len(failed_samples) / len(samples)
            if failure_rate > 0.3:
                filters_detected.append(f"High failure rate: {failure_rate:.1%}")
        
        # Analyze response time patterns for throttling
        response_times = [s.response_time for s in samples if s.response_time > 0]
        if response_times:
            avg_response = statistics.mean(response_times)
            if avg_response > 5.0:  # More than 5 seconds average
                filters_detected.append(f"Potential throttling: {avg_response:.1f}s avg response")
        
        # Content-based filter detection
        blocked_patterns = sum(1 for s in samples if s.anomaly_type == AnomalyType.BLOCKED)
        if blocked_patterns > len(samples) * 0.1:
            filters_detected.append("Active content filtering detected")
        
        return filters_detected
    
    async def _analyze_recent_patterns(self, samples: List[TrafficSample]) -> Dict[str, Any]:
        """Analyze recent pattern matches"""
        recent_patterns = defaultdict(int)
        
        for sample in samples[-100:]:  # Last 100 samples
            pattern_matches = await self.behavioral_analyst.pattern_matching(sample)
            for pattern_id in pattern_matches:
                recent_patterns[pattern_id] += 1
        
        return {
            'total_matches': sum(recent_patterns.values()),
            'most_common_patterns': dict(sorted(recent_patterns.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    async def _identify_emerging_threats(self, samples: List[TrafficSample]) -> List[Dict[str, Any]]:
        """Identify emerging threats and new patterns"""
        emerging_threats = []
        
        # Look for new anomaly patterns
        recent_anomalies = [s for s in samples[-50:] if s.anomaly_score > 0.7]
        if len(recent_anomalies) > 5:
            emerging_threats.append({
                'type': 'cluster_anomalies',
                'count': len(recent_anomalies),
                'risk': 'HIGH',
                'description': 'Cluster of high-anomaly traffic detected'
            })
        
        # Detect new traffic patterns
        if len(self.behavioral_analyst.patterns) > len(samples) // 100:
            emerging_threats.append({
                'type': 'pattern_proliferation',
                'count': len(self.behavioral_analyst.patterns),
                'risk': 'MEDIUM',
                'description': 'Unusual number of traffic patterns emerging'
            })
        
        return emerging_threats
    
    def _generate_security_recommendations(self, threat_level: int, 
                                         detected_filters: List[str],
                                         pattern_analysis: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []
        
        if threat_level >= 4:
            recommendations.extend([
                "Immediate proxy method rotation recommended",
                "Enable maximum encryption levels",
                "Consider temporary operation pause",
                "Review all recent traffic patterns"
            ])
        elif threat_level >= 3:
            recommendations.extend([
                "Increase monitoring frequency",
                "Implement additional authentication layers",
                "Review and update filter bypass strategies"
            ])
        
        if detected_filters:
            recommendations.append(
                f"Active filters detected: {', '.join(detected_filters)} - adjust strategies accordingly"
            )
        
        if pattern_analysis['emerging_threats']:
            recommendations.append(
                "Emerging threats detected - enhance security protocols"
            )
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get traffic analyzer status and statistics"""
        return {
            "samples_analyzed": len(self.traffic_samples),
            "patterns_detected": len(self.behavioral_analyst.patterns),
            "current_threat_level": self._get_current_threat_level(),
            "monitoring_active": self.is_monitoring,
            "recent_intelligence": len(self.intelligence_reports),
            "analysis_depth": self.analysis_depth
        }
    
    def _get_current_threat_level(self) -> int:
        """Get current threat level based on recent analysis"""
        if not self.intelligence_reports:
            return 1
        
        latest_report = self.intelligence_reports[-1]
        return latest_report.threat_level
