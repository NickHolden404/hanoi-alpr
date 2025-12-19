"""
Vietnamese license plate validation and formatting.
"""

import re
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class PlateValidator:
    """Validates and normalizes Vietnamese license plates."""
    
    # Vietnamese plate format patterns
    FORMATS = {
        'standard_car': re.compile(r'^\d{2}[A-Z]-\d{3}\.\d{2}$'),      # 29A-246.53
        'compact': re.compile(r'^\d{2}[A-Z]{2}-\d{4}$'),               # 20KM-161.20
        'motorcycle': re.compile(r'^\d{2}[A-Z]-\d{4}$'),               # 29D-1657
        'alternative': re.compile(r'^\d{2}[A-Z]{2}-\d{3}\.\d{2}$'),   # 22RL-102.20
    }
    
    # Character normalization (common OCR mistakes)
    CHAR_REPLACEMENTS = {
        'O': '0',
        'I': '1',
        'l': '1',
        'Z': '2',
        'S': '5',
        'G': '6',
        'B': '8',
    }
    
    def __init__(self, min_length: int = 7, max_length: int = 12,
                 custom_formats: Optional[List[str]] = None):
        """
        Initialize plate validator.
        
        Args:
            min_length: Minimum valid plate length
            max_length: Maximum valid plate length
            custom_formats: Additional regex patterns for valid plates
        """
        self.min_length = min_length
        self.max_length = max_length
        
        # Add custom formats if provided
        if custom_formats:
            for i, pattern in enumerate(custom_formats):
                self.FORMATS[f'custom_{i}'] = re.compile(pattern)
    
    def normalize(self, plate_text: str) -> str:
        """
        Normalize plate text (remove spaces, fix common OCR errors).
        
        Args:
            plate_text: Raw plate text from OCR
            
        Returns:
            Normalized plate text
        """
        # Remove whitespace
        plate = plate_text.strip().upper()
        plate = ''.join(plate.split())
        
        # Apply character replacements in numeric sections only
        # Split by hyphen to identify sections
        if '-' in plate:
            parts = plate.split('-')
            # First part: numbers + letter(s)
            # Second part: numbers (+ optional dot + numbers)
            
            # Normalize first part (keep letters as-is)
            first = parts[0]
            # Last 1-2 chars are letters, rest are numbers
            
            # Normalize second part (all should be numbers)
            if len(parts) > 1:
                second = parts[1]
                for old, new in self.CHAR_REPLACEMENTS.items():
                    second = second.replace(old, new)
                plate = f"{first}-{second}"
        
        return plate
    
    def is_valid_format(self, plate_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if plate matches any known Vietnamese format.
        
        Args:
            plate_text: Normalized plate text
            
        Returns:
            Tuple of (is_valid, format_name)
        """
        for format_name, pattern in self.FORMATS.items():
            if pattern.match(plate_text):
                return True, format_name
        
        return False, None
    
    def is_valid_length(self, plate_text: str) -> bool:
        """Check if plate length is within valid range."""
        length = len(plate_text)
        return self.min_length <= length <= self.max_length
    
    def validate(self, plate_text: str, confidence: float = 0.0) -> Tuple[bool, str]:
        """
        Validate plate text completely.
        
        Args:
            plate_text: Raw plate text from OCR
            confidence: OCR confidence score
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Normalize first
        normalized = self.normalize(plate_text)
        
        # Check length
        if not self.is_valid_length(normalized):
            return False, f"Invalid length: {len(normalized)}"
        
        # Check format
        is_valid, format_name = self.is_valid_format(normalized)
        if not is_valid:
            return False, "Does not match Vietnamese plate format"
        
        # All checks passed
        return True, f"Valid ({format_name})"
    
    def get_plate_type(self, plate_text: str) -> str:
        """
        Determine vehicle type from plate format.
        
        Args:
            plate_text: Normalized plate text
            
        Returns:
            Vehicle type ('car', 'motorcycle', or 'unknown')
        """
        is_valid, format_name = self.is_valid_format(plate_text)
        
        if not is_valid:
            return 'unknown'
        
        if format_name == 'motorcycle':
            return 'motorcycle'
        elif format_name in ['standard_car', 'compact', 'alternative']:
            return 'car'
        
        return 'unknown'


# Test function
if __name__ == "__main__":
    validator = PlateValidator()
    
    # Test cases
    test_plates = [
        "29A-246.53",      # Valid standard car
        "20KM-161.20",     # Valid compact
        "29D-1657",        # Valid motorcycle
        "29A 246 53",      # Valid but needs normalization
        "29A246.53",       # Missing hyphen
        "XX-123.45",       # Invalid prefix
        "29A-24.53",       # Too short
        "29A-2461.530",    # Too long
    ]
    
    print("Testing plate validator:\n")
    for plate in test_plates:
        normalized = validator.normalize(plate)
        is_valid, reason = validator.validate(plate)
        plate_type = validator.get_plate_type(normalized)
        
        print(f"Input:      {plate}")
        print(f"Normalized: {normalized}")
        print(f"Valid:      {is_valid} - {reason}")
        print(f"Type:       {plate_type}")
        print("-" * 50)
