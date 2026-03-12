// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IPAttestation
 * @notice On-chain timestamping contract for Cultural IP Securitization research data
 * @dev Deployed on Ethereum Sepolia Testnet (Chain ID: 11155111)
 * @author Xu Yuming, 2026
 */
contract IPAttestation {
    
    // ============================================================
    // Events
    // ============================================================
    event DataAttested(
        uint256 indexed attestationId,
        bytes32 indexed contentHash,
        string description,
        address attester,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    // ============================================================
    // State Variables
    // ============================================================
    address public immutable owner;
    uint256 public attestationCount;
    
    struct Attestation {
        uint256 id;
        bytes32 contentHash;    // SHA256 hash of the attested data
        string description;     // Human-readable description
        address attester;       // Address that submitted the attestation
        uint256 timestamp;      // Block timestamp
        uint256 blockNumber;    // Block number for additional verifiability
    }
    
    mapping(uint256 => Attestation) public attestations;
    mapping(bytes32 => uint256) public hashToId;  // Lookup by content hash
    
    // ============================================================
    // Constructor
    // ============================================================
    constructor() {
        owner = msg.sender;
        attestationCount = 0;
    }
    
    // ============================================================
    // Core Functions
    // ============================================================
    
    /**
     * @notice Attest a content hash on-chain
     * @param contentHash SHA256 hash of the data to attest (bytes32)
     * @param description Human-readable description of the attested data
     * @return attestationId The ID of the new attestation
     */
    function attest(
        bytes32 contentHash,
        string calldata description
    ) external returns (uint256 attestationId) {
        require(contentHash != bytes32(0), "Invalid content hash");
        require(bytes(description).length > 0, "Description required");
        require(hashToId[contentHash] == 0, "Hash already attested");
        
        attestationCount++;
        attestationId = attestationCount;
        
        attestations[attestationId] = Attestation({
            id: attestationId,
            contentHash: contentHash,
            description: description,
            attester: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number
        });
        
        hashToId[contentHash] = attestationId;
        
        emit DataAttested(
            attestationId,
            contentHash,
            description,
            msg.sender,
            block.timestamp,
            block.number
        );
        
        return attestationId;
    }
    
    /**
     * @notice Verify if a content hash has been attested
     * @param contentHash The hash to verify
     * @return exists Whether the hash has been attested
     * @return attestation The attestation details (if exists)
     */
    function verify(bytes32 contentHash) 
        external 
        view 
        returns (bool exists, Attestation memory attestation) 
    {
        uint256 id = hashToId[contentHash];
        if (id == 0) {
            return (false, Attestation(0, bytes32(0), "", address(0), 0, 0));
        }
        return (true, attestations[id]);
    }
    
    /**
     * @notice Get attestation by ID
     * @param attestationId The attestation ID
     * @return The attestation details
     */
    function getAttestation(uint256 attestationId) 
        external 
        view 
        returns (Attestation memory) 
    {
        require(attestationId > 0 && attestationId <= attestationCount, "Invalid ID");
        return attestations[attestationId];
    }
}
